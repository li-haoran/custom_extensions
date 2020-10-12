# EMD approximation module (based on auction algorithm)
# memory complexity: O(n)
# time complexity: O(n^2 * iter) 
# author: Minghua Liu

# input:
# p1, p2: [#batch, #points, 3]
# where p1 is the predicted point cloud and p2 is the ground truth point cloud 
# two point clouds should have same size and be normalized to [0, 1]
# #points should be a multiple of 1024
# #batch should be no greater than 512
# eps is a parameter which balances the error rate and the speed of convergence
# iters is the number of iteration
# we only calculate gradient for p1

# Output:
# dist: [#batch, #points],  sqrt(dist) -> L2 distance 
# assignment: [#batch, #points], index of the matched point in the ground truth point cloud
# the result is an approximation and the assignment is not guranteed to be a bijection


import torch
from torch import nn
from..functions.emd import emd_function


class emdModule(nn.Module):
    def __init__(self):
        super(emdModule, self).__init__()

    def forward(self, p1, p2, eps, iters):
        return emd_function(p1, p2, eps, iters)


        