#ifndef _EMD
#define _EMD

#include <vector>
#include <torch/extension.h>

//CUDA declarations
at::Tensor ApproxMatchForward(
    const at::Tensor p1,
    const at::Tensor p2);

at::Tensor MatchCostForward(
    const at::Tensor p1,
    const at::Tensor p2,
    const at::Tensor match);

std::vector<at::Tensor> MatchCostBackward(
    const at::Tensor grad_cost,
    const at::Tensor p1,
    const at::Tensor p2,
    const at::Tensor match);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("approxmatch_forward", &ApproxMatchForward,"ApproxMatch forward (CUDA)");
  m.def("matchcost_forward", &MatchCostForward,"MatchCost forward (CUDA)");
  m.def("matchcost_backward", &MatchCostBackward,"MatchCost backward (CUDA)");
}

#endif
