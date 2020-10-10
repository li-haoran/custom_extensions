/*!
 *****************
 * COPYRIGHT
 *
 * LICENSE
 *
 * author: haoran li

 */

// modify from
// https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/mmdetection/mmdet/ops/dcn/src/deform_conv_cuda_kernel.cu

#include <ATen/ATen.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <THC/THCAtomics.cuh>

using namespace at;

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
const int kMaxGridNum = 65535;
inline int GET_BLOCKS(const int N) {
  return std::min(kMaxGridNum, (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
}

template <typename scalar_t>
__device__ scalar_t offsetAccumulate_im2col_bilinear(
    const scalar_t* bottom_data,
    const int data_width,
    const int height,
    const int width,
    scalar_t h,
    scalar_t w) {
  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  scalar_t lh = h - h_low;
  scalar_t lw = w - w_low;
  scalar_t hh = 1 - lh, hw = 1 - lw;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = bottom_data[h_low * data_width + w_low];
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = bottom_data[h_low * data_width + w_high];
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = bottom_data[h_high * data_width + w_low];
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = bottom_data[h_high * data_width + w_high];

  scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename scalar_t>
__device__ scalar_t get_gradient_weight(
    scalar_t argmax_h,
    scalar_t argmax_w,
    const int h,
    const int w,
    const int height,
    const int width) {
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 ||
      argmax_w >= width) {
    // empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  scalar_t weight = 0;
  if (h == argmax_h_low && w == argmax_w_low)
    weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
  if (h == argmax_h_low && w == argmax_w_high)
    weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
  if (h == argmax_h_high && w == argmax_w_low)
    weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
  if (h == argmax_h_high && w == argmax_w_high)
    weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
  return weight;
}

template <typename scalar_t>
__device__ scalar_t get_coordinate_weight(
    scalar_t argmax_h,
    scalar_t argmax_w,
    const int height,
    const int width,
    const scalar_t ratio_h,
    const scalar_t ratio_w,
    const scalar_t* u_data,
    const scalar_t* v_data,
    const scalar_t top_grad_h,
    const scalar_t top_grad_w,
    const int data_width,
    int bp_dir) {
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 ||
      argmax_w >= width) {
    // empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  scalar_t weight = 0;

  if (bp_dir == 0) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_w_low + 1 - argmax_w) *
          (u_data[argmax_h_low * data_width + argmax_w_low] * top_grad_h +
           v_data[argmax_h_low * data_width + argmax_w_low] * ratio_h /
               ratio_w * top_grad_w);
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += -1 * (argmax_w - argmax_w_low) *
          (u_data[argmax_h_low * data_width + argmax_w_high] * top_grad_h +
           v_data[argmax_h_low * data_width + argmax_w_high] * ratio_h /
               ratio_w * top_grad_w);
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += (argmax_w_low + 1 - argmax_w) *
          (u_data[argmax_h_high * data_width + argmax_w_low] * top_grad_h +
           v_data[argmax_h_high * data_width + argmax_w_low] * ratio_h /
               ratio_w * top_grad_w);
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_w - argmax_w_low) *
          (u_data[argmax_h_high * data_width + argmax_w_high] * top_grad_h +
           v_data[argmax_h_high * data_width + argmax_w_high] * ratio_h /
               ratio_w * top_grad_w);
  } else if (bp_dir == 1) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_h_low + 1 - argmax_h) *
          (u_data[argmax_h_low * data_width + argmax_w_low] * ratio_w /
               ratio_h * top_grad_h +
           v_data[argmax_h_low * data_width + argmax_w_low] * top_grad_w);
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += (argmax_h_low + 1 - argmax_h) *
          (u_data[argmax_h_low * data_width + argmax_w_high] * ratio_w /
               ratio_h * top_grad_h +
           v_data[argmax_h_low * data_width + argmax_w_high] * top_grad_w);
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += -1 * (argmax_h - argmax_h_low) *
          (u_data[argmax_h_high * data_width + argmax_w_low] * ratio_w /
               ratio_h * top_grad_h +
           v_data[argmax_h_high * data_width + argmax_w_low] * top_grad_w);
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_h - argmax_h_low) *
          (u_data[argmax_h_high * data_width + argmax_w_high] * ratio_w /
               ratio_h * top_grad_h +
           v_data[argmax_h_high * data_width + argmax_w_high] * top_grad_w);
  }

  return weight;
}

template <typename scalar_t>
__global__ void offsetAccumulate_im2col_gpu_kernel(
    const int n,
    const scalar_t* data_dcn,
    const scalar_t* data_target,
    const int num_target,
    const int num_offset,
    const int height_in,
    const int width_in,
    const int height_out,
    const int width_out,
    const scalar_t ratio_h,
    const scalar_t ratio_w,
    scalar_t* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    // index index of output matrix
    const int w_col = index % width_out;
    const int h_col = (index / width_out) % height_out;
    const int c_points = index / width_out / height_out;
    const int c_col = c_points * 2;
    const int c_target = c_points / num_offset;
    const int c_dcn = c_points % num_offset;

    scalar_t* data_col_ptr =
        data_col + (c_col * height_out + h_col) * width_out + w_col;
    // const scalar_t* data_im_ptr = data_im + ((b_col * num_channels + c_im) *
    // height + h_in) * width + w_in;
    const scalar_t* data_dcn_ptr = data_dcn + c_dcn * 2 * height_in * width_in;
    const scalar_t* data_target_ptr =
        data_target + (c_target * 2 * height_out + h_col) * width_out + w_col;

    const int data_target_h_ptr = 0;
    const int data_target_w_ptr = height_out * width_out;
    const scalar_t offset_h = data_target_ptr[data_target_h_ptr];
    const scalar_t offset_w = data_target_ptr[data_target_w_ptr];
    scalar_t val_h = static_cast<scalar_t>(0);
    scalar_t val_w = static_cast<scalar_t>(0);

    const scalar_t h_im = (h_col + offset_h) * ratio_h;
    const scalar_t w_im = (w_col + offset_w) * ratio_w;
    if (h_im > -1 && w_im > -1 && h_im < height_in && w_im < width_in) {
      const scalar_t* data_dcn_ptr_h = data_dcn_ptr;
      val_h = offsetAccumulate_im2col_bilinear(
          data_dcn_ptr_h, width_in, height_in, width_in, h_im, w_im);
      const scalar_t* data_dcn_ptr_w = data_dcn_ptr + height_in * width_in;
      val_w = offsetAccumulate_im2col_bilinear(
          data_dcn_ptr_w, width_in, height_in, width_in, h_im, w_im);
    }
    scalar_t* data_col_ptr_h = data_col_ptr;
    *data_col_ptr_h = val_h / ratio_h + offset_h;
    scalar_t* data_col_ptr_w = data_col_ptr + height_out * width_out;
    *data_col_ptr_w = val_w / ratio_w + offset_w;
  }
}

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
    at::Tensor data_col) {
  // num_axes should be smaller than block size
  // todo: check parallel_imgs is correctly passed in
  int num_kernels = num_target * num_offset * height_out * width_out;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_col.type(), "offsetAccumulate_im2col_gpu", ([&] {
        const scalar_t* data_dcn_ = data_dcn.data<scalar_t>();
        const scalar_t* data_target_ = data_target.data<scalar_t>();
        const scalar_t ratio_h_ = static_cast<scalar_t>(ratio_h);
        const scalar_t ratio_w_ = static_cast<scalar_t>(ratio_w);
        scalar_t* data_col_ = data_col.data<scalar_t>();

        offsetAccumulate_im2col_gpu_kernel<<<
            GET_BLOCKS(num_kernels),
            CUDA_NUM_THREADS>>>(
            num_kernels,
            data_dcn_,
            data_target_,
            num_target,
            num_offset,
            height_in,
            width_in,
            height_out,
            width_out,
            ratio_h_,
            ratio_w_,
            data_col_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in offsetAccumulate_im2col: %s\n", cudaGetErrorString(err));
  }
}

template <typename scalar_t>
__global__ void offsetAccumulate_col2im_gpu_kernel(
    const int n,
    const scalar_t* grad_output,
    const scalar_t* data_target,
    const int num_target,
    const int num_offset,
    const int height_in,
    const int width_in,
    const int height_out,
    const int width_out,
    const scalar_t ratio_h,
    const scalar_t ratio_w,
    scalar_t* grad_dcn_offset) {
  CUDA_KERNEL_LOOP(index, n) {
    const int w_col = index % width_out;
    const int h_col = (index / width_out) % height_out;
    const int m_col = index / width_out / height_out % num_offset;
    const int n_col = index / width_out / height_out / num_offset;
    // compute the start and end of the output

    const int grad_output_index_base =
        (2 * m_col * height_out + h_col) * width_out + w_col;

    const int plane_num = height_out * width_out;
    const int step = 2 * n_col * plane_num;

    const scalar_t* data_grad_output_ptr = grad_output + grad_output_index_base;

    const scalar_t* data_target_ptr = data_target + h_col * width_out + w_col;

    scalar_t* data_grad_dcn_offset_ptr =
        grad_dcn_offset + 2 * m_col * height_in * width_in;

    const int index_h = step * num_offset;
    const int index_w = step * num_offset + plane_num;
    const scalar_t cur_top_grad_h = data_grad_output_ptr[index_h];
    const scalar_t cur_top_grad_w = data_grad_output_ptr[index_w];

    const scalar_t cur_target_h = data_target_ptr[step];
    const scalar_t cur_target_w = data_target_ptr[step + plane_num];

    const scalar_t cur_inv_h_data = (h_col + cur_target_h) * ratio_h;
    const scalar_t cur_inv_w_data = (w_col + cur_target_w) * ratio_w;

    const int cur_h = (int)cur_inv_h_data;
    const int cur_w = (int)cur_inv_w_data;

    for (int dy = -2; dy <= 2; dy++) {
      for (int dx = -2; dx <= 2; dx++) {
        if (cur_h + dy >= 0 && cur_h + dy < height_in && cur_w + dx >= 0 &&
            cur_w + dx < width_in && abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
            abs(cur_inv_w_data - (cur_w + dx)) < 1) {
          int cur_bottom_grad_pos_h = (cur_h + dy) * width_in + cur_w + dx;
          scalar_t weight = get_gradient_weight(
              cur_inv_h_data,
              cur_inv_w_data,
              cur_h + dy,
              cur_w + dx,
              height_in,
              width_in);

          scalar_t this_grad_h = 1.0 / ratio_h * weight * cur_top_grad_h;
          atomicAdd(
              data_grad_dcn_offset_ptr + cur_bottom_grad_pos_h, this_grad_h);

          int cur_bottom_grad_pos_w =
              height_in * width_in + cur_bottom_grad_pos_h;
          scalar_t this_grad_w = 1.0 / ratio_w * weight * cur_top_grad_w;
          atomicAdd(
              data_grad_dcn_offset_ptr + cur_bottom_grad_pos_w, this_grad_w);
        }
      }
    }
  }
}

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
    at::Tensor grad_dcn_offset) {
  int num_kernels = num_target * num_offset * height_out * width_out;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.type(), "offsetAccumulate_col2im_gpu", ([&] {
        const scalar_t* grad_output_ = grad_output.data<scalar_t>();
        const scalar_t* data_target_ = data_target.data<scalar_t>();
        scalar_t* grad_dcn_offset_ = grad_dcn_offset.data<scalar_t>();
        const scalar_t ratio_h_ = static_cast<scalar_t>(ratio_h);
        const scalar_t ratio_w_ = static_cast<scalar_t>(ratio_w);

        offsetAccumulate_col2im_gpu_kernel<<<
            GET_BLOCKS(num_kernels),
            CUDA_NUM_THREADS>>>(
            num_kernels,
            grad_output_,
            data_target_,
            num_target,
            num_offset,
            height_in,
            width_in,
            height_out,
            width_out,
            ratio_h_,
            ratio_w_,
            grad_dcn_offset_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf(
        "error in offsetAccumulate_col2im: %s\n",
        cudaGetErrorString(err));
  }
}

template <typename scalar_t>
__global__ void offsetAccumulate_col2im_coord_gpu_kernel(
    const int n,
    const scalar_t* grad_output,
    const scalar_t* data_dcn,
    const scalar_t* data_target,
    const int num_target,
    const int num_offset,
    const int height_in,
    const int width_in,
    const int height_out,
    const int width_out,
    const scalar_t ratio_h,
    const scalar_t ratio_w,
    scalar_t* grad_target_offset) {
  CUDA_KERNEL_LOOP(index, n) {
    const int w_col = index % width_out;
    const int h_col = (index / width_out) % height_out;
    const int m_col = index / width_out / height_out % num_offset;
    const int n_col = index / width_out / height_out / num_offset;
    // compute the start and end of the output

    const int grad_output_index_base =
        (2 * m_col * height_out + h_col) * width_out + w_col;

    const int plane_num = height_out * width_out;
    const int step = 2 * n_col * plane_num;

    const scalar_t* data_grad_output_ptr = grad_output + grad_output_index_base;
    const scalar_t* data_target_ptr = data_target + h_col * width_out + w_col;
    const int index_h = step * num_offset;
    const int index_w = step * num_offset + plane_num;
    const scalar_t cur_top_grad_h = data_grad_output_ptr[index_h];
    const scalar_t cur_top_grad_w = data_grad_output_ptr[index_w];

    const scalar_t cur_target_h = data_target_ptr[step];
    const scalar_t cur_target_w = data_target_ptr[step + plane_num];

    scalar_t cur_inv_h_data = (h_col + cur_target_h) * ratio_h;
    scalar_t cur_inv_w_data = (w_col + cur_target_w) * ratio_w;

    const scalar_t* u_data_ptr = data_dcn + 2 * m_col * height_in * width_in;
    const scalar_t* v_data_ptr =
        data_dcn + (2 * m_col + 1) * height_in * width_in;

    scalar_t* data_grad_target_offset_ptr = grad_target_offset +
        2 * n_col * height_out * width_out + h_col * width_out + w_col;
    if (cur_inv_h_data <= -1 || cur_inv_w_data <= -1 ||
        cur_inv_h_data >= height_in || cur_inv_w_data >= width_in) {
      cur_inv_h_data = cur_inv_w_data = -2;
    }
    const scalar_t weight_h = get_coordinate_weight(
        cur_inv_h_data,
        cur_inv_w_data,
        height_in,
        width_in,
        ratio_h,
        ratio_w,
        u_data_ptr,
        v_data_ptr,
        cur_top_grad_h,
        cur_top_grad_w,
        width_in,
        0);

    scalar_t val_h = cur_top_grad_h + weight_h;
    atomicAdd(data_grad_target_offset_ptr, val_h);

    const scalar_t weight_w = get_coordinate_weight(
        cur_inv_h_data,
        cur_inv_w_data,
        height_in,
        width_in,
        ratio_h,
        ratio_w,
        u_data_ptr,
        v_data_ptr,
        cur_top_grad_h,
        cur_top_grad_w,
        width_in,
        1);

    scalar_t val_w = cur_top_grad_w + weight_w;
    atomicAdd(data_grad_target_offset_ptr + height_out * width_out, val_w);
  }
}

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
    at::Tensor grad_target_offset) {
  int num_kernels = num_target * num_offset * height_out * width_out;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.type(), "offsetAccumulate_col2im_coord_gpu", ([&] {
        const scalar_t* grad_output_ = grad_output.data<scalar_t>();
        const scalar_t* data_target_ = data_target.data<scalar_t>();
        const scalar_t* data_dcn_ = data_dcn.data<scalar_t>();
        const scalar_t ratio_h_ = static_cast<scalar_t>(ratio_h);
        const scalar_t ratio_w_ = static_cast<scalar_t>(ratio_w);
        scalar_t* grad_target_offset_ = grad_target_offset.data<scalar_t>();

        offsetAccumulate_col2im_coord_gpu_kernel<<<
            GET_BLOCKS(num_kernels),
            CUDA_NUM_THREADS>>>(
            num_kernels,
            grad_output_,
            data_dcn_,
            data_target_,
            num_target,
            num_offset,
            height_in,
            width_in,
            height_out,
            width_out,
            ratio_h_,
            ratio_w_,
            grad_target_offset_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf(
        "error in offsetAccumulate_col2im_coord: %s\n",
        cudaGetErrorString(err));
  }
}
