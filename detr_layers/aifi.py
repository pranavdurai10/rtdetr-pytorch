class AIFI(nn.Module):
    def __init__(self, in_channels, num_repeats):
        super(AIFI, self).__init__()
        # Initialize the layers for AIFI
        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            self.layers.append(nn.Linear(in_channels, in_channels))

    def forward(self, x):
        # Implement the forward pass for AIFI
        for layer in self.layers:
            x = layer(x)
        return x