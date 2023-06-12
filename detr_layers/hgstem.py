class HGStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HGStem, self).__init__()
        # Initialize the layers for HGStem
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Implement the forward pass for HGStem
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x