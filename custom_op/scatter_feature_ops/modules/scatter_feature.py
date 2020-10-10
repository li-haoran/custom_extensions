import math

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

from ..functions.scatter_feature import scatter_feature


class ScatterFeaturePack(nn.Module):

    def __init__(self,):
        super(ScatterFeaturePack, self).__init__()

    def forward(self, feature, sample_offsets,batch_index):
        return scatter_feature(feature,sample_offsets,batch_index)
