'''
///////////////////////////////////////////////////////////////////////////
Code written by Pranav Durai on 09.06.2023 @ 21:19:34

Model: RTDeTR-X (Realtime Detection Transformer)

Framework: PyTorch 2.0

Architecture: 

backbone:
  # [from, repeats, module, args]
  - [-1, 1, HGStem, [32, 64]]  # 0-P2/4
  - [-1, 6, HGBlock, [64, 128, 3]]  # stage 1

  - [-1, 1, DWConv, [128, 3, 2, 1, False]]  # 2-P3/8
  - [-1, 6, HGBlock, [128, 512, 3]]
  - [-1, 6, HGBlock, [128, 512, 3, False, True]]   # 4-stage 2

  - [-1, 1, DWConv, [512, 3, 2, 1, False]]  # 5-P3/16
  - [-1, 6, HGBlock, [256, 1024, 5, True, False]]  # cm, c2, k, light, shortcut
  - [-1, 6, HGBlock, [256, 1024, 5, True, True]]
  - [-1, 6, HGBlock, [256, 1024, 5, True, True]]
  - [-1, 6, HGBlock, [256, 1024, 5, True, True]]
  - [-1, 6, HGBlock, [256, 1024, 5, True, True]]  # 10-stage 3

  - [-1, 1, DWConv, [1024, 3, 2, 1, False]]  # 11-P4/32
  - [-1, 6, HGBlock, [512, 2048, 5, True, False]]
  - [-1, 6, HGBlock, [512, 2048, 5, True, True]]  # 13-stage 4

head:
  - [-1, 1, Conv, [384, 1, 1, None, 1, 1, False]]  # 14 input_proj.2
  - [-1, 1, AIFI, [2048, 8]]
  - [-1, 1, Conv, [384, 1, 1]]   # 16, Y5, lateral_convs.0

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [10, 1, Conv, [384, 1, 1, None, 1, 1, False]]  # 18 input_proj.1
  - [[-2, -1], 1, Concat, [1]]
  - [-1, 3, RepC3, [384]]  # 20, fpn_blocks.0
  - [-1, 1, Conv, [384, 1, 1]]   # 21, Y4, lateral_convs.1

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [4, 1, Conv, [384, 1, 1, None, 1, 1, False]]  # 23 input_proj.0
  - [[-2, -1], 1, Concat, [1]]  # cat backbone P4
  - [-1, 3, RepC3, [384]]    # X3 (25), fpn_blocks.1

  - [-1, 1, Conv, [384, 3, 2]]   # 26, downsample_convs.0
  - [[-1, 21], 1, Concat, [1]]  # cat Y4
  - [-1, 3, RepC3, [384]]    # F4 (28), pan_blocks.0

  - [-1, 1, Conv, [384, 3, 2]]   # 29, downsample_convs.1
  - [[-1, 16], 1, Concat, [1]]  # cat Y5
  - [-1, 3, RepC3, [384]]    # F5 (31), pan_blocks.1

  - [[25, 28, 31], 1, RTDETRDecoder, [nc]]  # Detect(P3, P4, P5)

///////////////////////////////////////////////////////////////////////////
'''

# Import necessary libraries
import torch
import torch.nn as nn

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


class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(DWConv, self).__init__()
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


class AIFI(nn.Module):
    def __init__(self, in_channels, num_repeats):
        super(AIFI, self).__init__()
        # Initialize the layers for AIFI
        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            self.layers.append(nn.Linear(in_channels, in_channels))

    def forward(self, x):
        # Implement the forward pass for AIFI
        for layer in self.layers:
            x = layer(x)
        return x


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


class Concat(nn.Module):
    def __init__(self, indices, dim):
        super(Concat, self).__init__()
        # Initialize the layers for Concat
        self.indices = indices
        self.dim = dim

    def forward(self, x):
        # Implement the forward pass for Concat
        concatenated = torch.cat([x[i] for i in self.indices], dim=self.dim)
        return concatenated


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


class RTDETRDecoder(nn.Module):
    def __init__(self, num_classes):
        super(RTDETRDecoder, self).__init__()
        # Initialize the layers for RTDETRDecoder
        self.num_classes = num_classes

    def forward(self, x):
        # Implement the forward pass for RTDETRDecoder
        return x


class RTDeTRX(nn.Module):
    def __init__(self, num_classes):
        super(RTDeTRX, self).__init__()

        self.backbone = nn.Sequential(
            HGStem(3, 64),                        # 0-P2/4
            HGBlock(64, 64, 128, 6),              # stage 1

            DWConv(128, 128, kernel_size=3, stride=2, padding=1, bias=False),  # 2-P3/8
            HGBlock(128, 128, 512, 6),
            HGBlock(128, 128, 512, 6),            # 4-stage 2

            DWConv(512, 512, kernel_size=3, stride=2, padding=1, bias=False),  # 5-P3/16
            HGBlock(256, 256, 1024, 6),           # cm, c2, k, light, shortcut
            HGBlock(256, 256, 1024, 6),
            HGBlock(256, 256, 1024, 6),
            HGBlock(256, 256, 1024, 6),
            HGBlock(256, 256, 1024, 6),           # 10-stage 3

            DWConv(1024, 1024, kernel_size=3, stride=2, padding=1, bias=False),  # 11-P4/32
            HGBlock(512, 512, 2048, 6),
            HGBlock(512, 512, 2048, 6),           # 13-stage 4
        )

        self.head = nn.Sequential(
            Conv(384, 384, kernel_size=1, stride=1, padding=0, bias=False),  # 14 input_proj.2
            AIFI(2048, 8),
            Conv(384, 384, kernel_size=1, stride=1, padding=0),  # 16, Y5, lateral_convs.0

            nn.Upsample(scale_factor=2, mode='nearest'),
            Conv(384, 384, kernel_size=1, stride=1, padding=0, bias=False),  # 18 input_proj.1
            Concat([-2, -1], dim=1),
            RepC3(384),                                 # 20, fpn_blocks.0
            Conv(384, 384, kernel_size=1, stride=1, padding=0),  # 21, Y4, lateral_convs.1

            nn.Upsample(scale_factor=2, mode='nearest'),
            Conv(384, 384, kernel_size=1, stride=1, padding=0, bias=False),  # 23 input_proj.0
            Concat([-2, -1], dim=1),                    # cat backbone P4
            RepC3(384),                                 # X3 (25), fpn_blocks.1

            Conv(384, 384, kernel_size=3, stride=2, padding=0),  # 26, downsample_convs.0
            Concat([-1, 21], dim=1),                    # cat Y4
            RepC3(384),                                 # F4 (28), pan_blocks.0

            Conv(384, 384, kernel_size=3, stride=2, padding=0),  # 29, downsample_convs.1
            Concat([-1, 16], dim=1),                    # cat Y5
            RepC3(384),                                 # F5 (31), pan_blocks.1

            RTDETRDecoder(num_classes)                  # Detect(P3, P4, P5)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

# Create an instance of RTDeTR
model = RTDeTRX(num_classes=80)