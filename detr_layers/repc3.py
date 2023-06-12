class RepC3(nn.Module):
    def __init__(self, in_channels):
        super(RepC3, self).__init__()
        # Initialize the layers for RepC3
        self.layers = nn.Sequential(
            Conv(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            Conv(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            Conv(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Implement the forward pass for RepC3
        return self.layers(x)