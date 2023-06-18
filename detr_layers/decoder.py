class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 attn_dropout=None, act_dropout=None, normalize_before=False):
        super(TransformerDecoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=attn_dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=attn_dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(act_dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = getattr(F, activation)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, pos_embed=None, query_pos_embed=None):
        tgt_mask = torch.tensor(tgt_mask, dtype=torch.bool) if tgt_mask is not None else None

        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        tgt = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask)[0]
        tgt = residual + self.dropout1(tgt)
        if not self.normalize_before:
            tgt = self.norm1(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)
        q = self.with_pos_embed(tgt, query_pos_embed)
        k = self.with_pos_embed(memory, pos_embed)
        tgt = self.cross_attn(q, k, value=memory, attn_mask=memory_mask)[0]
        tgt = residual + self.dropout2(tgt)
        if not self.normalize_before:
            tgt = self.norm2(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm3(tgt)
        tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = residual + self.dropout3(tgt)
        if not self.normalize_before:
            tgt = self.norm3(tgt)
        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, pos_embed=None, query_pos_embed=None):
        tgt_mask = torch.tensor(tgt_mask, dtype=torch.bool) if tgt_mask is not None else None

        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                           pos_embed=pos_embed, query_pos_embed=query_pos_embed)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class RTDETRDecoder(nn.Module):
    def __init__(self, num_classes):
        super(RTDETRDecoder, self).__init__()
        '''
        RTDETRDecoder implementation.

        params:
                num_classes (int): Number of classes in the dataset.
        '''
        # Initialize the layers for RTDETRDecoder
        self.num_classes = num_classes
        d_model = 512  # Set the appropriate value for d_model
        nhead = 8  # Set the appropriate value for nhead
        dim_feedforward = 2048  # Set the appropriate value for dim_feedforward
        dropout = 0.1  # Set the appropriate value for dropout
        activation = "relu"  # Set the appropriate value for activation

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=6)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None, pos_embed=None, query_pos_embed=None):
    # Implement the forward pass for RTDETRDecoder
    output = self.transformer_decoder(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                      pos_embed=pos_embed, query_pos_embed=query_pos_embed)
    return output