// modify from
// https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/mmdetection/mmdet/ops/dcn/src/deform_conv_cuda.c

#include <torch/extension.h>

#include <cmath>
#include <vector>

void scatter_img2inst(
    const at::Tensor data_feature,
    const at::Tensor data_sample_offsets,
    const at::Tensor data_batch_index,
    const int num_instance,
    const int num_points,
    const int channel_out,
    const int height_out,
    const int width_out,
    at::Tensor data_col);

void scatter_inst2img(
    const at::Tensor data_feature,
    const at::Tensor data_sample_offsets,
    const at::Tensor data_batch_index,
    const int num_instance,
    const int num_points,
    const int channel_out,
    const int height_out,
    const int width_out,
    at::Tensor grad_feature,
    const at::Tensor grad_output);

void coord_inst2img(
    const at::Tensor data_feature,
    const at::Tensor data_sample_offsets,
    const at::Tensor data_batch_index,
    const int num_instance,
    const int num_points,
    const int channel_out,
    const int height_out,
    const int width_out,
    at::Tensor grad_sample_offsets,
    const at::Tensor grad_output);

void shape_check(at::Tensor feature, at::Tensor sample_offsets) {
  TORCH_CHECK(
      feature.ndimension() == 4,
      "4D feature tensor (b,c,H,W) expected, "
      "but got: %s",
      feature.ndimension());

  TORCH_CHECK(feature.is_contiguous(), "feature tensor has to be contiguous");

  TORCH_CHECK(
      sample_offsets.ndimension() == 3,
      "3D sample_offsets tensor (n,p,2) expected, "
      "but got: %s",
      sample_offsets.ndimension());

  TORCH_CHECK(
      sample_offsets.is_contiguous(),
      "sample_offsets tensor has to be contiguous");
}

void scatter_feature_forward_cuda(
    at::Tensor feature,
    at::Tensor sample_offsets,
    at::Tensor batch_index,
    at::Tensor output) {
  shape_check(feature, sample_offsets);

  feature = feature.contiguous();
  sample_offsets = sample_offsets.contiguous();
  batch_index = batch_index.contiguous();

  const int batch = feature.size(0);
  const int channel_out = feature.size(1);
  const int height_out = feature.size(2);
  const int width_out = feature.size(3);

  const int n_instance = sample_offsets.size(0);
  const int n_points = sample_offsets.size(1);

  // resize output
  output = output.zero_();

  scatter_img2inst(
      feature,
      sample_offsets,
      batch_index,
      n_instance,
      n_points,
      channel_out,
      height_out,
      width_out,
      output);
}

void scatter_feature_backward_cuda(
    at::Tensor feature,
    at::Tensor sample_offsets,
    at::Tensor batch_index,
    at::Tensor grad_feature,
    at::Tensor grad_sample_offsets,
    at::Tensor grad_output) {
  TORCH_CHECK(feature.is_contiguous(), "feature tensor has to be contiguous");
  TORCH_CHECK(
      sample_offsets.is_contiguous(),
      "sample_offsets tensor has to be contiguous");

  const int batch = feature.size(0);
  const int channel_out = feature.size(1);
  const int height_out = feature.size(2);
  const int width_out = feature.size(3);

  const int n_instance = sample_offsets.size(0);
  const int n_points = sample_offsets.size(1);

  scatter_inst2img(
      feature,
      sample_offsets,
      batch_index,
      n_instance,
      n_points,
      channel_out,
      height_out,
      width_out,
      grad_feature,
      grad_output);

  coord_inst2img(
      feature,
      sample_offsets,
      batch_index,
      n_instance,
      n_points,
      channel_out,
      height_out,
      width_out,
      grad_sample_offsets,
      grad_output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "scatter_feature_forward_cuda",
      &scatter_feature_forward_cuda,
      "scatter feature forward(CUDA)");
  m.def(
      "scatter_feature_backward_cuda",
      &scatter_feature_backward_cuda,
      "scatter feature backward(CUDA)");
}
