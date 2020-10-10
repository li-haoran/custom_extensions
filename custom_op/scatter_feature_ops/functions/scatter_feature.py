import torch
from torch.autograd import Function
from torch.nn.modules.utils import _pair

from .. import scatter_feature_cuda


class ScatterFeatureFunction(Function):

    @staticmethod
    def forward(ctx,
                feature,
                sample_offsets, 
                batch_index,
                ):#dilation ==1 for first try

        if sample_offsets is not None and sample_offsets.dim() != 3:
            raise ValueError(
                "Expected 3D tensor as input, got {}D tensor instead.".format(
                    sample_offsets.dim()))

        

        ctx.save_for_backward(feature,sample_offsets,batch_index)


        output = feature.new_empty(
            ScatterFeatureFunction._output_size(feature,sample_offsets))

        if not sample_offsets.is_cuda:
            raise NotImplementedError
        else:
           
            assert (sample_offsets.size(2) % 2) == 0, 'sample_offsets channels must be even number'
            assert batch_index.size(0)==sample_offsets.size(0),'the num_instance mismath'
            scatter_feature_cuda.scatter_feature_forward_cuda(
                feature,sample_offsets, batch_index,output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        feature,sample_offsets,batch_index = ctx.saved_tensors

        grad_feature = grad_sample_offsets =  None

        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                grad_sample_offsets = torch.zeros_like(sample_offsets)
                grad_feature = torch.zeros_like(feature)
                scatter_feature_cuda.scatter_feature_backward_cuda(
                feature,sample_offsets,batch_index,grad_feature,grad_sample_offsets,grad_output)
                

        return (grad_feature,grad_sample_offsets,None)

    @staticmethod
    def _output_size(feature,sample_offsets):

        b,c,h,w=feature.size()
        n=sample_offsets.size(0)
        
        output_size = (n,c,h,w)
        return output_size




scatter_feature = ScatterFeatureFunction.apply
