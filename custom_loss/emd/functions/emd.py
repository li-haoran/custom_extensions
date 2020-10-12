# EMD approximation module (based on auction algorithm)
# memory complexity: O(n)
# time complexity: O(n^2 * iter) 
# author: Minghua Liu

# Input:
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
from torch.autograd import Function
from .. import emd




class emdFunction(Function):
    @staticmethod
    def forward(ctx, p1, p2, eps, iters):

        batchsize, n, _ = p1.size()
        _, m, _ = p2.size()

        assert(n == m)
        assert(p1.size(0) == p2.size(0))
        assert(batchsize <= 512)

        p1 = p1.contiguous().float().cuda()
        p2 = p2.contiguous().float().cuda()
        dist = torch.zeros(batchsize, n, device='cuda').contiguous()
        assignment = torch.zeros(batchsize, n, device='cuda', dtype=torch.int32).contiguous() - 1
        assignment_inv = torch.zeros(batchsize, m, device='cuda', dtype=torch.int32).contiguous() - 1
        price = torch.zeros(batchsize, m, device='cuda').contiguous()
        bid = torch.zeros(batchsize, n, device='cuda', dtype=torch.int32).contiguous()
        bid_increments = torch.zeros(batchsize, n, device='cuda').contiguous()
        max_increments = torch.zeros(batchsize, m, device='cuda').contiguous()
        unass_idx = torch.zeros(batchsize * n, device='cuda', dtype=torch.int32).contiguous()
        max_idx = torch.zeros(batchsize * m, device='cuda', dtype=torch.int32).contiguous()
        unass_cnt = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()
        unass_cnt_sum = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()
        cnt_tmp = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()

        emd.forward(p1, p2, dist, assignment, price, assignment_inv, bid, bid_increments, max_increments, unass_idx, unass_cnt, unass_cnt_sum, cnt_tmp, max_idx, eps, iters)

        ctx.save_for_backward(p1, p2, assignment)
        return dist, assignment

    @staticmethod
    def backward(ctx, graddist, gradidx):
        p1, p2, assignment = ctx.saved_tensors
        graddist = graddist.contiguous()

        gradp1 = torch.zeros(p1.size(), device='cuda').contiguous()
        gradp2 = torch.zeros(p2.size(), device='cuda').contiguous()

        emd.backward(p1, p2, gradp1, graddist, assignment)
        return gradp1, gradp2, None, None

emd_function=emdFunction.apply