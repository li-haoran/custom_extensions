import cv2
import numpy as np
import time
from PyTorchEMD.modules.emd import EarthMoverDistance as pytorchemd
from PyTorchEMD.functions.emd import earth_mover_distance
import torch.nn as nn
import torch

class opencvemd(nn.Module):
    def __init__(self):
        super(opencvemd,self).__init__()

    def forward(self,p1,p2):

        up1=p1.unsqueeze(2)
        up2=p2.unsqueeze(1)
        b,m,_=p1.size()

        distance_map=torch.sum((up1-up2)**2,dim=3)

        index=p1.new_zeros((b,m,1)).long()



        for i in range(b):
            _, _, flow = cv2.EMD(p1[i].detach().cpu().numpy(), p2[i].detach().cpu().numpy(), cv2.DIST_USER, distance_map[i].detach().cpu().numpy())
            flow=np.argmax(flow,axis=1)
            print('opencvemd:',np.sort(flow))
            index[i,:,0]=index.new_tensor(flow)
        index=index.repeat((1,1,2))

        new_p2=torch.gather(p2,1,index)

        cost=torch.sum((p1-new_p2)**2,dim=2)
        cost=torch.sum(cost,dim=1)
        
        return cost

def test_op():
    opemd=opencvemd()

    p1=torch.rand(1,729,2).float().cuda()
    p2=torch.rand(1,729,2).float().cuda()
    cost1=opemd(p1,p2)

    cost2=earth_mover_distance(p1,p2)

    print(cost1,cost2)
            


def benchmark(k=1000):

    
    ptemd=pytorchemd()
    opemd=opencvemd()


    start_time=time.time()*1000
    for i in range(k):
        p1=torch.rand(512,729,2).float().cuda()
        p2=torch.rand(512,729,2).float().cuda()
        p1.requires_grad=True
        p2.requires_grad=True
        cost=ptemd(p1,p2)

        loss=torch.mean(cost)
        loss.backward()
    end_time =time.time()*1000
    print('pytorch emd benmark for {} round cost: {} ms'.format(k,end_time-start_time))

    start_time=time.time()*1000
    for i in range(k):
        p1=torch.rand(36,729,2).float().cuda()
        p2=torch.rand(36,729,2).float().cuda()
        p1.requires_grad=True
        p2.requires_grad=True

        cost=opemd(p1,p2)

        loss=torch.mean(cost)
        loss.backward()
    end_time =time.time()*1000
    print('opencv emd benmark for {} round cost: {} ms'.format(k,end_time-start_time))


if __name__=='__main__':
    # benchmark(10)
    test_op()
