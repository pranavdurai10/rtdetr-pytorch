class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(Conv, self).__init__()
        # Initialize the layers for Conv
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Implement the forward pass for Conv
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x