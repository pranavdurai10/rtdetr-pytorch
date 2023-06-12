class RTDETRDecoder(nn.Module):
    def __init__(self, num_classes):
        super(RTDETRDecoder, self).__init__()
        # Initialize the layers for RTDETRDecoder
        self.num_classes = num_classes

    def forward(self, x):
        # Implement the forward pass for RTDETRDecoder
        return x