import torch
from torch.autograd import Function
from torch.nn.modules.utils import _pair

from .. import points_collect_cuda


class PointsCollectFunction(Function):

    @staticmethod
    def forward(ctx,
                target_offset, # target offset here use 0,0 for root level
                dcn_offset, # deformbale offset shold plus kernel offset
                ):#dilation ==1 for first try

        if target_offset is not None and target_offset.dim() != 4:
            raise ValueError(
                "Expected 4D tensor as input, got {}D tensor instead.".format(
                    target_offset.dim()))

        ctx.save_for_backward(target_offset,dcn_offset)

        output = target_offset.new_empty(
            PointsCollectFunction._output_size(target_offset,dcn_offset))


        if not target_offset.is_cuda:
            raise NotImplementedError
        else:
           
            assert (target_offset.size(1) % 2) == 0, 'target_offset channels must be even number'
            assert (dcn_offset.size(1) % 2) == 0, 'dcn_offset channels must be even number'
            points_collect_cuda.points_collect_forward_cuda(
                target_offset,dcn_offset, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        target_offset,dcn_offset = ctx.saved_tensors

        grad_target_offset = grad_dcn_offset =  None

        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                grad_target_offset = torch.zeros_like(target_offset)
                grad_dcn_offset = torch.zeros_like(dcn_offset)
                points_collect_cuda.points_collect_backward_cuda(
                target_offset,dcn_offset, grad_target_offset,grad_dcn_offset,grad_output)
                

        return (grad_target_offset,grad_dcn_offset)

    @staticmethod
    def _output_size(target_offset,dcn_offset):

        b,c,h,w=target_offset.size()
        c1=dcn_offset.size(1)//2
        
        output_size = (b,c*c1,h,w)
        return output_size




points_collect = PointsCollectFunction.apply
