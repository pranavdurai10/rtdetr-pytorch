class Concat(nn.Module):
    def __init__(self, indices, dim):
        super(Concat, self).__init__()
        # Initialize the layers for Concat
        self.indices = indices
        self.dim = dim

    def forward(self, x):
        # Implement the forward pass for Concat
        concatenated = torch.cat([x[i] for i in self.indices], dim=self.dim)
        return concatenated