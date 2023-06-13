class HGBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, num_repeats, *args):
        super(HGBlock, self).__init__()
        '''
        HGBlock implementation.

        params: 
                in_channels (int): Number of input channels for the block.
                mid_channels (int): Number of intermediate channels within the block.
                out_channels (int): Number of output channels for the block.
                num_repeats (int): Number of times the block should be repeated.
                *args: Variable-length argument list for any additional arguments.
        
        variables:         
                ModuleList (list): Holds sub-modules in a list. Used to store the layers. Used to store the layer of HGBlock.
        '''
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