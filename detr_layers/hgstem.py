class HGStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HGStem, self).__init__()
        '''
        HGSTEM implementation.
        
        params:
                in_channels (int): Number of input channels. 
                out_channels (int): Number of output channels.
                kernel_size (int or tuple): Size of the convolving kernel.
                stride (int or tuple): Stride of the convolution.
                padding (int or tuple): Padding added to all the sides of the input.
                inplace (bool): If set to True, the input will be modified in-placed. Default: False.
        '''
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Implement the forward pass for HGStem
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x