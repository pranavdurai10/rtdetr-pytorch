class AIFI(nn.Module):
    def __init__(self, in_channels, num_repeats):
        super(AIFI, self).__init__()
        '''
        AIFI implementation.

        params:
                in_channels (int): Number of input channels for AIFI.
                num_repeats (int): Number of times the block should be repeated.
        variables:         
                ModuleList (list): Holds sub-modules in a list. Used to store the layers. Used to store the layer of AIFI.
        '''
        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            self.layers.append(nn.Linear(in_channels, in_channels))

    def forward(self, x):
        # Implement the forward pass for AIFI
        for layer in self.layers:
            x = layer(x)
        return x