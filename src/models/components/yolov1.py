import torch
import torch.nn as nn

class YOLOv1(nn.Module):
    def __init__(
        self, 
        net: nn.Module,
        fcs: nn.Module,
        in_channels:int = 3, 
        split_size: int = 7, 
        num_boxes: int = 2, 
        num_classes: int = 20,
        **kwargs
    ) -> None:
        super().__init__()
        
        self.net = net
        self.fcs = fcs  
        self.in_channels = in_channels
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
    
    def forward(self, x):
        x = self.net(x)    
        return self.fcs(torch.flatten(x, start_dim=1))

    