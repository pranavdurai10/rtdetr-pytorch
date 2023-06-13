'''
///////////////////////////////////////////////////////////////////////////
Code written by Pranav Durai on 06.06.2023 @ 21:50:12

Model: RTDeTR-L (Realtime Detection Transformer)

Framework: PyTorch 2.0

Architecture: 

backbone:
  # [from, repeats, module, args]
  - [-1, 1, HGStem, [32, 48]]  # 0-P2/4
  - [-1, 6, HGBlock, [48, 128, 3]]  # stage 1

  - [-1, 1, DWConv, [128, 3, 2, 1, False]]  # 2-P3/8
  - [-1, 6, HGBlock, [96, 512, 3]]   # stage 2

  - [-1, 1, DWConv, [512, 3, 2, 1, False]]  # 4-P3/16
  - [-1, 6, HGBlock, [192, 1024, 5, True, False]]  # cm, c2, k, light, shortcut
  - [-1, 6, HGBlock, [192, 1024, 5, True, True]]
  - [-1, 6, HGBlock, [192, 1024, 5, True, True]]  # stage 3

  - [-1, 1, DWConv, [1024, 3, 2, 1, False]]  # 8-P4/32
  - [-1, 6, HGBlock, [384, 2048, 5, True, False]]  # stage 4

head:
  - [-1, 1, Conv, [256, 1, 1, None, 1, 1, False]]  # 10 input_proj.2
  - [-1, 1, AIFI, [1024, 8]]
  - [-1, 1, Conv, [256, 1, 1]]   # 12, Y5, lateral_convs.0

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [7, 1, Conv, [256, 1, 1, None, 1, 1, False]]  # 14 input_proj.1
  - [[-2, -1], 1, Concat, [1]]
  - [-1, 3, RepC3, [256]]  # 16, fpn_blocks.0
  - [-1, 1, Conv, [256, 1, 1]]   # 17, Y4, lateral_convs.1

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [3, 1, Conv, [256, 1, 1, None, 1, 1, False]]  # 19 input_proj.0
  - [[-2, -1], 1, Concat, [1]]  # cat backbone P4
  - [-1, 3, RepC3, [256]]    # X3 (21), fpn_blocks.1

  - [-1, 1, Conv, [256, 3, 2]]   # 22, downsample_convs.0
  - [[-1, 17], 1, Concat, [1]]  # cat Y4
  - [-1, 3, RepC3, [256]]    # F4 (24), pan_blocks.0

  - [-1, 1, Conv, [256, 3, 2]]   # 25, downsample_convs.1
  - [[-1, 12], 1, Concat, [1]]  # cat Y5
  - [-1, 3, RepC3, [256]]    # F5 (27), pan_blocks.1

  - [[21, 24, 27], 1, RTDETRDecoder, [nc]]  # Detect(P3, P4, P5)

///////////////////////////////////////////////////////////////////////////
'''

# Import necessary libraries
import torch
import torch.nn as nn

from detr_layers.hgstem import HGStem
from detr_layers.hgblock import HGBlock
from detr_layers.dwconv import DWConv
from detr_layers.aifi import AIFI
from detr_layers.conv import Conv
from detr_layers.concat import Concat
from detr_layers.repc3 import RepC3
from detr_layers.decoder import RTDETRDecoder

class RTDeTRL(nn.Module):
    def __init__(self, num_classes, scales):
        super(RTDeTRL, self).__init__()
        self.num_classes = num_classes

        self.backbone = nn.ModuleList([
            HGStem(3, int(48 * scales[1])),
            HGBlock(int(48 * scales[1]), int(128 * scales[1]), 6),
            DWConv(int(128 * scales[1]), 3, 2, 1, False),
            HGBlock(int(96 * scales[1]), int(512 * scales[1]), 6),
            DWConv(int(512 * scales[1]), 3, 2, 1, False),
            HGBlock(int(192 * scales[1]), int(1024 * scales[1]), 5, True, False),
            HGBlock(int(192 * scales[1]), int(1024 * scales[1]), 5, True, True),
            HGBlock(int(192 * scales[1]), int(1024 * scales[1]), 5, True, True),
            DWConv(int(1024 * scales[1]), 3, 2, 1, False),
            HGBlock(int(384 * scales[1]), int(2048 * scales[1]), 5, True, False)
        ])

        self.head = nn.ModuleList([
            Conv(int(256 * scales[1]), 256, 1, False),
            AIFI(int(1024 * scales[1]), 8),
            Conv(256, 256, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            Conv(256, 256, 1, False),
            nn.Upsample(scale_factor=2, mode='nearest'),
            Conv(256, 256, 1, False),
            nn.Upsample(scale_factor=2, mode='nearest'),
            Conv(256, 256, 3, stride=2),
            Conv(256 + 256, 256, 3, stride=2),
            Conv(256 + 256, 256, 3, stride=2),
            RTDETRDecoder([256, 256, 256], self.num_classes)
        ])

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [3, 5, 7, 9]:
                features.append(x)

        for i, layer in enumerate(self.head):
            if isinstance(layer, nn.Upsample):
                x = layer(x)
                x = torch.cat((x, features.pop()), dim=1)
            else:
                x = layer(x)

        return x


# Create an instance of the RTDeTRL model
model = RTDeTRL(num_classes=80, scales={'l': [1.00, 1.00, 1024]})