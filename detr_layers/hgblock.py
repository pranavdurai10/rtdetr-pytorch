class HGBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, num_repeats, *args):
        super(HGBlock, self).__init__()
        # Initialize the layers for HGBlock
        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

    def forward(self, x):
        # Implement the forward pass for HGBlock
        for layer in self.layers:
            x = layer(x)
        return x