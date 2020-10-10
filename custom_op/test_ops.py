

#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import torch
import torch.nn as nn
from torch.autograd import gradcheck,grad
import  matplotlib.pyplot as plt
import numpy as np

from points_collection_ops.modules.points_collect import PointsCollectPack
from points_collection_ops.functions.points_collect import points_collect
from scatter_feature_ops.functions.scatter_feature import scatter_feature

N, inC, inH, inW = 2, 18, 5,5
tgC=4



def check_zero_offset():
    
    PCP=PointsCollectPack(3,1,1).cuda()

    meshgrid=PCP.anchor

    dcn_offset = torch.rand(N, inC, inH, inW).cuda()
    target_offset = torch.zeros(N, tgC, inH, inW).cuda()

    output = PCP(target_offset,dcn_offset)
    d = (dcn_offset+meshgrid - output[:,:inC,:,:]).abs().max()
    if d < 1e-10:
        print('Zero offset passed')
    else:
        print('Zero offset failed')
        print(dcn_offset)
        print(output)

def check_gradient_pc():
    N=2
    dcn_offset = torch.rand(N, inC, inH, inW).double().cuda()
    dcn_offset.requires_grad = True
    dcn_offset.retain_grad()

    target_offset = torch.rand(N, tgC, inH, inW).double().cuda()*2+1
    # offset.data.zero_()
    # offset.data -= 0.5
    target_offset.requires_grad = True
    target_offset.retain_grad()

    output=points_collect(target_offset,dcn_offset)

    # gradinputlist=grad(output,(target_offset,dcn_offset),
    # grad_outputs=torch.ones_like(output),retain_graph=True)
    # print(gradinputlist[0][0,:,4,4],gradinputlist[1][0,:,4,4])
    # print(dcn_offset[0,:,4,4],target_offset[0,:,4,4])
    # print(output[0,:,4,4])

    print('check_gradient_pc: ',
          gradcheck(points_collect, (target_offset,dcn_offset),
                    ))#eps=1e-3, atol=1e-4, rtol=1e-2

def plot_according_to_point(im, source_points, center):

    plt.figure()
    plt.imshow(im)
    y = source_points[:,0]
    x = source_points[:,1]

    plt.scatter(x,y,c='r',)

    plt.scatter(center[1],center[0],c='g')
    plt.show()



def example_pcp():
    N=1
    PCP=PointsCollectPack(3,1,1).cuda()

    meshgrid=PCP.anchor

    dcn_offset = torch.rand(N, inC, 64, 64).cuda()*2-1
    target_offset = torch.zeros(N, tgC, 64, 64).cuda()

    output=PCP(target_offset,dcn_offset)

    output=PCP(output,dcn_offset)

    center=(32,32)#(y,x)
    source_points=output.cpu().numpy()[0,:,center[0],center[1]]
    source_points=np.reshape(source_points,(-1,2))+32

    print(source_points)

    im=np.ones((64,64))
    plot_according_to_point(im,source_points,center)


def check_scatter():
    N=2
    inC=16
    feature = torch.rand(N, inC, 5, 5).double().cuda()
    # feature[0]=0.5
    # feature[1]=1
    sample_offsets=torch.rand(5,2,2).double().cuda()*4
    # sample_offsets[:]=0.5
    batch_index=torch.tensor([0,1,0,1,1]).double().cuda()

    feature.requires_grad = True
    feature.retain_grad()
    sample_offsets.requires_grad = True
    sample_offsets.retain_grad()
    # batch_index.requires_grad = True
    # batch_index.retain_grad()



    # output=scatter_feature(feature,sample_offsets,batch_index)
    # print(feature,sample_offsets,batch_index,output)
    print('check_gradient_sf: ',
          gradcheck(scatter_feature, (feature,sample_offsets,batch_index),
                    ))#eps=1e-3, atol=1e-4, rtol=1e-2




if __name__ == '__main__':

    # example_pcp()

    # check_zero_offset()
    # # a=input('num:')

    # check_gradient_pc()
    check_scatter()