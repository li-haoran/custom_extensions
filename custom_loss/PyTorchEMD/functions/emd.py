import torch
from .. import emd_cuda


class EarthMoverDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p1, p2):
        p1 = p1.contiguous()
        p2 = p2.contiguous()
        assert p1.is_cuda and p2.is_cuda, "Only support cuda currently."
        match = emd_cuda.approxmatch_forward(p1, p2)
        cost = emd_cuda.matchcost_forward(p1, p2, match)
        _,id=torch.max(match,dim=2)
        id,_=torch.sort(id,dim=1)
        print('pytorchemd:',id[0])
        ctx.save_for_backward(p1, p2, match)
        return cost

    @staticmethod
    def backward(ctx, grad_cost):
        p1, p2, match = ctx.saved_tensors
        grad_cost = grad_cost.contiguous()
        grad_p1, grad_p2 = emd_cuda.matchcost_backward(grad_cost, p1, p2, match)
        return grad_p1, grad_p2


earth_mover_distance=EarthMoverDistanceFunction.apply

