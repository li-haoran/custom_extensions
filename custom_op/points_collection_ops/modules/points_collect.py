import math

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

from ..functions.points_collect import points_collect


class PointsCollectPack(nn.Module):

    def __init__(self,
                 kernel_size=3,
                 pad=1,
                 dilation=1,
                 ):
        super(PointsCollectPack, self).__init__()

        self.kernel_size = _pair(kernel_size)
        self.pad = _pair(pad)
        self.dilation = _pair(dilation)


        pos_shift_w = torch.tensor([x*self.dilation[1]-self.pad[1] for x in range(self.kernel_size[1])])
        pos_shift_h = torch.tensor([x*self.dilation[0]-self.pad[0] for x in range(self.kernel_size[0])])
        grid=torch.meshgrid(pos_shift_h,pos_shift_w)
        grid=torch.stack(grid,dim=2)

        grid=grid.view(1,self.kernel_size[1]*self.kernel_size[0]*2,1,1)
        self.register_buffer('anchor',grid)


    def forward(self, target_offset, dcn_offset):

        assert dcn_offset.size(1)==self.anchor.size(1),'dcn offset channles wrong!'
        dcn_offset=dcn_offset+self.anchor
             
        return points_collect(target_offset, dcn_offset)
