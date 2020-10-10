import torch
import torch.nn as nn
from ..functions.emd import earth_mover_distance


class EarthMoverDistance(nn.Module):
    def __init__(self,):
        super(EarthMoverDistance,self).__init__()

    def forward(self,p1,p2):
        return earth_mover_distance(p1,p2)


