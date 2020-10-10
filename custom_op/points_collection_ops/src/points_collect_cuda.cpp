// modify from
// https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/mmdetection/mmdet/ops/dcn/src/deform_conv_cuda.c

#include <torch/extension.h>

#include <cmath>
#include <vector>

void offsetAccumulate_im2col(
    const at::Tensor data_dcn,
    const at::Tensor data_target,
    const int num_target,
    const int num_offset,
    const int height_out,
    const int width_out,
    const int height_in,
    const int width_in,
    float ratio_h,
    float ratio_w,
    at::Tensor data_col);

void offsetAccumulate_col2im(
    const at::Tensor grad_output,
    const at::Tensor data_target,
    const int num_target,
    const int num_offset,
    const int height_out,
    const int width_out,
    const int height_in,
    const int width_in,
    float ratio_h,
    float ratio_w,
    at::Tensor grad_dcn_offset);

void offsetAccumulate_col2im_coord(
    const at::Tensor grad_output,
    const at::Tensor data_dcn,
    const at::Tensor data_target,
    const int num_target,
    const int num_offset,
    const int height_out,
    const int width_out,
    const int height_in,
    const int width_in,
    float ratio_h,
    float ratio_w,
    at::Tensor grad_target_offset);

void shape_check(at::Tensor target_offset, at::Tensor dcn_offset) {
  TORCH_CHECK(
      target_offset.ndimension() == 4,
      "4D target_offset tensor (b,c,H,W) expected, "
      "but got: %s",
      target_offset.ndimension());

  TORCH_CHECK(
      target_offset.is_contiguous(),
      "target_offset tensor has to be contiguous");

  TORCH_CHECK(
      dcn_offset.ndimension() == 4,
      "4D target_offset tensor (b,c,H,W) expected, "
      "but got: %s",
      dcn_offset.ndimension());

  TORCH_CHECK(
      dcn_offset.is_contiguous(), "dcn_offset tensor has to be contiguous");
}

void points_collect_forward_cuda(
    at::Tensor target_offset,
    at::Tensor dcn_offset,
    at::Tensor output) {
  shape_check(target_offset, dcn_offset);

  target_offset = target_offset.contiguous();
  dcn_offset = dcn_offset.contiguous();

  // todo: assert batch dividable by im2col_step

  const int batch = target_offset.size(0);
  const int num_target = target_offset.size(1) / 2;
  const int height_out = target_offset.size(2);
  const int width_out = target_offset.size(3);

  const int width_in = dcn_offset.size(3);
  const int height_in = dcn_offset.size(2);
  const int num_offset = dcn_offset.size(1) / 2;
  const int channels_out = num_target * num_offset * 2;

  const float ratio_h = 1.0 * height_in / height_out;
  const float ratio_w = 1.0 * width_in / width_out;

  // resize output
  output = output.view({batch, channels_out, height_out, width_out}).zero_();

  for (int b = 0; b < batch; b++) {
    offsetAccumulate_im2col(
        dcn_offset[b],
        target_offset[b],
        num_target,
        num_offset,
        height_out,
        width_out,
        height_in,
        width_in,
        ratio_h,
        ratio_w,
        output[b]);
    // channels_out means the target points size.
  }
}

void points_collect_backward_cuda(
    at::Tensor target_offset,
    at::Tensor dcn_offset,
    at::Tensor grad_target_offset,
    at::Tensor grad_dcn_offset,
    at::Tensor grad_output) {
  TORCH_CHECK(
      target_offset.is_contiguous(),
      "target_offset tensor has to be contiguous");
  TORCH_CHECK(
      dcn_offset.is_contiguous(), "dcn_offset tensor has to be contiguous");

  const int batch = target_offset.size(0);
  const int num_target = target_offset.size(1) / 2;
  const int height_out = target_offset.size(2);
  const int width_out = target_offset.size(3);

  const int width_in = dcn_offset.size(3);
  const int height_in = dcn_offset.size(2);
  const int num_offset = dcn_offset.size(1) / 2;

  const float ratio_h = 1.0 * height_in / height_out;
  const float ratio_w = 1.0 * width_in / width_out;

  grad_output = grad_output.view(
      {grad_output.size(0),
       grad_output.size(1),
       grad_output.size(2),
       grad_output.size(3)});

  for (int b = 0; b < batch; b++) {
    // gradient w.r.t. input dcn offset data
    offsetAccumulate_col2im(
        grad_output[b],
        target_offset[b],
        num_target,
        num_offset,
        height_out,
        width_out,
        height_in,
        width_in,
        ratio_h,
        ratio_w,
        grad_dcn_offset[b]);

    offsetAccumulate_col2im_coord(
        grad_output[b],
        dcn_offset[b],
        target_offset[b],
        num_target,
        num_offset,
        height_out,
        width_out,
        height_in,
        width_in,
        ratio_h,
        ratio_w,
        grad_target_offset[b]);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "points_collect_forward_cuda",
      &points_collect_forward_cuda,
      "points collect forward (CUDA)");
  m.def(
      "points_collect_backward_cuda",
      &points_collect_backward_cuda,
      "points collect backward (CUDA)");
}
