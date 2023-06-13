class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(DWConv, self).__init__()
        '''
        DWConv implementation.

        params:
                in_channels (int): Number of input channels for DWConv.
                out_channels (int): Number of output channels for DWConv.
                kernel_size (int or tuple): Size of the convolving kernel.
                stride (int or tuple): Stride of the kernel.
                padding (int or tuple): Addition of padding around the kernel.
                bias (bool): If True, adds a learnable bias to the output.
        '''
        # Initialize the layers for DWConv
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=in_channels, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Implement the forward pass for DWConv
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x