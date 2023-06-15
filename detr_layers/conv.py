class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(Conv, self).__init__()
        '''
        Conv implementation.

        params:
                in_channels (int): Number of input channels.
                out_channels (int): Number of output channels.
                kernel_size (int or tuple): Size of the convolving kernel.
                stride (int or tuple): Stride of the kernel.
                padding (int): Addition of padding around the kernel.
                bias (boolean): If True, adds a learnable bias to the output.
        '''
        # Initialize the layers for Conv
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.SiLU(inplace=True)

    def forward(self, x):
        # Implement the forward pass for Conv
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x