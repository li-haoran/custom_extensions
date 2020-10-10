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
__device__ scalar_t bilinear(
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
    const scalar_t* im_data,
    const int data_width,
    const int bp_dir) {
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
          im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += -1 * (argmax_w - argmax_w_low) *
          im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += (argmax_w_low + 1 - argmax_w) *
          im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_w - argmax_w_low) *
          im_data[argmax_h_high * data_width + argmax_w_high];
  } else if (bp_dir == 1) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_h_low + 1 - argmax_h) *
          im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += (argmax_h_low + 1 - argmax_h) *
          im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += -1 * (argmax_h - argmax_h_low) *
          im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_h - argmax_h_low) *
          im_data[argmax_h_high * data_width + argmax_w_high];
  }

  return weight;
}

template <typename scalar_t>
__global__ void scatter_img2inst_gpu_kernel(
    const int n,
    const scalar_t* data_feature_,
    const scalar_t* data_sample_offsets_,
    const scalar_t* data_batch_index_,
    const int num_instance,
    const int num_points,
    const int channel_out,
    const int height_out,
    const int width_out,
    scalar_t* data_col_) {
  CUDA_KERNEL_LOOP(index, n) {
    // index index of output matrix
    const int p = index % num_points;
    const int c = index / num_points % channel_out;
    const int o = index / num_points / channel_out;

    const int b = data_batch_index_[o];

    const int data_offset_h_ptr = (o * num_points + p) * 2;
    const int data_offset_w_ptr = (o * num_points + p) * 2 + 1;
    const scalar_t h_im = data_sample_offsets_[data_offset_h_ptr];
    const scalar_t w_im = data_sample_offsets_[data_offset_w_ptr];
    const int data_feature_index =
        b * channel_out * height_out * width_out + c * height_out * width_out;
    const int data_col_index =
        o * channel_out * height_out * width_out + c * height_out * width_out;
    scalar_t* data_col_ptr = data_col_ + data_col_index;
    if (h_im > -1 && h_im < height_out && w_im > -1 && w_im < width_out) {
      scalar_t val = bilinear(
          data_feature_ + data_feature_index,
          width_out,
          height_out,
          width_out,
          h_im,
          w_im);

      int h_low = floor(h_im);
      int w_low = floor(w_im);
      int h_high = h_low + 1;
      int w_high = w_low + 1;
      // ll
      if (h_low >= 0 && w_low >= 0) {
        scalar_t weight = (h_low + 1 - h_im) * (w_low + 1 - w_im);
        atomicAdd(data_col_ptr + h_low * width_out + w_low, weight * val);
      }

      // lh
      if (h_low >= 0 && w_high <= width_out - 1) {
        scalar_t weight = (h_low + 1 - h_im) * (w_im + 1 - w_high);
        atomicAdd(data_col_ptr + h_low * width_out + w_high, weight * val);
      }
      // hl
      if (h_high <= height_out - 1 && w_low >= 0) {
        scalar_t weight = (h_im + 1 - h_high) * (w_low + 1 - w_im);
        atomicAdd(data_col_ptr + h_high * width_out + w_low, weight * val);
      }
      // hh
      if (h_high <= height_out - 1 && w_high <= width_out - 1) {
        scalar_t weight = (h_im + 1 - h_high) * (w_im + 1 - w_high);
        atomicAdd(data_col_ptr + h_high * width_out + w_high, weight * val);
      }
    }
  }
}

void scatter_img2inst(
    const at::Tensor data_feature,
    const at::Tensor data_sample_offsets,
    const at::Tensor data_batch_index,
    const int num_instance,
    const int num_points,
    const int channel_out,
    const int height_out,
    const int width_out,
    at::Tensor data_col) {
  // num_axes should be smaller than block size
  // todo: check parallel_imgs is correctly passed in
  int num_kernels = num_instance * channel_out * num_points;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_col.type(), "scatter_img2inst_gpu", ([&] {
        const scalar_t* data_feature_ = data_feature.data<scalar_t>();
        const scalar_t* data_sample_offsets_ =
            data_sample_offsets.data<scalar_t>();
        const scalar_t* data_batch_index_ = data_batch_index.data<scalar_t>();
        scalar_t* data_col_ = data_col.data<scalar_t>();

        scatter_img2inst_gpu_kernel<<<
            GET_BLOCKS(num_kernels),
            CUDA_NUM_THREADS>>>(
            num_kernels,
            data_feature_,
            data_sample_offsets_,
            data_batch_index_,
            num_instance,
            num_points,
            channel_out,
            height_out,
            width_out,
            data_col_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in scatter_img2inst: %s\n", cudaGetErrorString(err));
  }
}

template <typename scalar_t>
__global__ void scatter_inst2img_gpu_kernel(
    const int n,
    const scalar_t* data_feature_,
    const scalar_t* data_sample_offsets_,
    const scalar_t* data_batch_index_,
    const int num_instance,
    const int num_points,
    const int channel_out,
    const int height_out,
    const int width_out,
    scalar_t* grad_feature_,
    const scalar_t* grad_output_) {
  CUDA_KERNEL_LOOP(index, n) {
    // index index of output matrix
    const int p = index % num_points;
    const int c = index / num_points % channel_out;
    const int o = index / num_points / channel_out;

    const int b = data_batch_index_[o];

    const int data_offset_h_index = (o * num_points + p) * 2;
    const int data_offset_w_index = (o * num_points + p) * 2 + 1;
    const scalar_t h_im = data_sample_offsets_[data_offset_h_index];
    const scalar_t w_im = data_sample_offsets_[data_offset_w_index];

    const int grad_output_index =
        o * channel_out * height_out * width_out + c * height_out * width_out;
    const scalar_t* grad_output_ptr = grad_output_ + grad_output_index;
    const int feature_index =
        b * channel_out * height_out * width_out + c * height_out * width_out;
    scalar_t* grad_feature_ptr = grad_feature_ + feature_index;

    if (h_im > -1 && h_im < height_out && w_im > -1 && w_im < width_out) {
      int h_low = floor(h_im);
      int w_low = floor(w_im);
      int h_high = h_low + 1;
      int w_high = w_low + 1;

      scalar_t mid_grad_feature = bilinear(
          grad_output_ptr, width_out, height_out, width_out, h_im, w_im);

      // ll
      if (h_low >= 0 && w_low >= 0) {
        scalar_t weight = (h_low + 1 - h_im) * (w_low + 1 - w_im);
        atomicAdd(
            grad_feature_ptr + h_low * width_out + w_low,
            weight * mid_grad_feature);
      }

      // lh
      if (h_low >= 0 && w_high <= width_out - 1) {
        scalar_t weight = (h_low + 1 - h_im) * (w_im + 1 - w_high);
        atomicAdd(
            grad_feature_ptr + h_low * width_out + w_high,
            weight * mid_grad_feature);
      }
      // hl
      if (h_high <= height_out - 1 && w_low >= 0) {
        scalar_t weight = (h_im + 1 - h_high) * (w_low + 1 - w_im);
        atomicAdd(
            grad_feature_ptr + h_high * width_out + w_low,
            weight * mid_grad_feature);
      }
      // hh
      if (h_high <= height_out - 1 && w_high <= width_out - 1) {
        scalar_t weight = (h_im + 1 - h_high) * (w_im + 1 - w_high);
        atomicAdd(
            grad_feature_ptr + h_high * width_out + w_high,
            weight * mid_grad_feature);
      }
    }
  }
}

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
    const at::Tensor grad_output) {
  int num_kernels = num_instance * channel_out * num_points;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.type(), "scatter_inst2img_gpu", ([&] {
        const scalar_t* grad_output_ = grad_output.data<scalar_t>();
        const scalar_t* data_feature_ = data_feature.data<scalar_t>();
        const scalar_t* data_sample_offsets_ =
            data_sample_offsets.data<scalar_t>();
        const scalar_t* data_batch_index_ = data_batch_index.data<scalar_t>();
        scalar_t* grad_feature_ = grad_feature.data<scalar_t>();

        scatter_inst2img_gpu_kernel<<<
            GET_BLOCKS(num_kernels),
            CUDA_NUM_THREADS>>>(
            num_kernels,
            data_feature_,
            data_sample_offsets_,
            data_batch_index_,
            num_instance,
            num_points,
            channel_out,
            height_out,
            width_out,
            grad_feature_,
            grad_output_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in scatter_inst2img: %s\n", cudaGetErrorString(err));
  }
}

template <typename scalar_t>
__global__ void coord_inst2img_gpu_kernel(
    const int n,
    const scalar_t* data_feature_,
    const scalar_t* data_sample_offsets_,
    const scalar_t* data_batch_index_,
    const int num_instance,
    const int num_points,
    const int channel_out,
    const int height_out,
    const int width_out,
    scalar_t* grad_sample_offsets_,
    const scalar_t* grad_output_) {
  CUDA_KERNEL_LOOP(index, n) {
    // index index of output matrix
    const int p = index % num_points;
    const int c = index / num_points % channel_out;
    const int o = index / num_points / channel_out;

    const int b = data_batch_index_[o];

    const int data_offset_h_index = (o * num_points + p) * 2;
    const int data_offset_w_index = (o * num_points + p) * 2 + 1;
    const scalar_t h_im = data_sample_offsets_[data_offset_h_index];
    const scalar_t w_im = data_sample_offsets_[data_offset_w_index];

    const int grad_output_index =
        o * channel_out * height_out * width_out + c * height_out * width_out;
    const scalar_t* grad_output_ptr = grad_output_ + grad_output_index;
    const int feature_index =
        b * channel_out * height_out * width_out + c * height_out * width_out;
    const scalar_t* data_feature_ptr = data_feature_ + feature_index;

    if (h_im > -1 && h_im < height_out && w_im > -1 && w_im < width_out) {
      scalar_t h_grad = 0;
      scalar_t w_grad = 0;

      scalar_t h_weight = get_coordinate_weight(
          h_im, w_im, height_out, width_out, data_feature_ptr, width_out, 0);
      scalar_t w_weight = get_coordinate_weight(
          h_im, w_im, height_out, width_out, data_feature_ptr, width_out, 1);

      int h_low = floor(h_im);
      int w_low = floor(w_im);
      int h_high = h_low + 1;
      int w_high = w_low + 1;

      scalar_t mid_feature = bilinear(
          data_feature_ptr, width_out, height_out, width_out, h_im, w_im);

      // ll
      if (h_low >= 0 && w_low >= 0) {
        scalar_t weight = (h_low + 1 - h_im) * (w_low + 1 - w_im);
        scalar_t h_coor_weight = -1 * (w_low + 1 - w_im);
        scalar_t w_coor_weight = -1 * (h_low + 1 - h_im);

        h_grad +=
            ((h_coor_weight * mid_feature + h_weight * weight) *
             grad_output_ptr[h_low * width_out + w_low]);
        w_grad +=
            ((w_coor_weight * mid_feature + w_weight * weight) *
             grad_output_ptr[h_low * width_out + w_low]);
      }

      // lh
      if (h_low >= 0 && w_high <= width_out - 1) {
        scalar_t weight = (h_low + 1 - h_im) * (w_im + 1 - w_high);
        scalar_t h_coor_weight = -1 * (w_im + 1 - w_high);
        scalar_t w_coor_weight = (h_low + 1 - h_im);

        h_grad +=
            ((h_coor_weight * mid_feature + h_weight * weight) *
             grad_output_ptr[h_low * width_out + w_high]);
        w_grad +=
            ((w_coor_weight * mid_feature + w_weight * weight) *
             grad_output_ptr[h_low * width_out + w_high]);
      }
      // hl
      if (h_high <= height_out - 1 && w_low >= 0) {
        scalar_t weight = (h_im + 1 - h_high) * (w_low + 1 - w_im);
        scalar_t h_coor_weight = (w_low + 1 - w_im);
        scalar_t w_coor_weight = -1 * (h_im + 1 - h_high);

        h_grad +=
            ((h_coor_weight * mid_feature + h_weight * weight) *
             grad_output_ptr[h_high * width_out + w_low]);
        w_grad +=
            ((w_coor_weight * mid_feature + w_weight * weight) *
             grad_output_ptr[h_high * width_out + w_low]);
      }
      // hh
      if (h_high <= height_out - 1 && w_high <= width_out - 1) {
        scalar_t weight = (h_im + 1 - h_high) * (w_im + 1 - w_high);
        scalar_t h_coor_weight = (w_im + 1 - w_high);
        scalar_t w_coor_weight = (h_im + 1 - h_high);

        h_grad +=
            ((h_coor_weight * mid_feature + h_weight * weight) *
             grad_output_ptr[h_high * width_out + w_high]);
        w_grad +=
            ((w_coor_weight * mid_feature + w_weight * weight) *
             grad_output_ptr[h_high * width_out + w_high]);
      }

      atomicAdd(grad_sample_offsets_+data_offset_h_index,h_grad);
      atomicAdd(grad_sample_offsets_+data_offset_w_index,w_grad);
    }
  }
}

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
    const at::Tensor grad_output) {
  int num_kernels = num_instance * channel_out * num_points;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.type(), "coord_inst2img_gpu", ([&] {
        const scalar_t* grad_output_ = grad_output.data<scalar_t>();
        const scalar_t* data_feature_ = data_feature.data<scalar_t>();
        const scalar_t* data_sample_offsets_ =
            data_sample_offsets.data<scalar_t>();
        const scalar_t* data_batch_index_ = data_batch_index.data<scalar_t>();
        scalar_t* grad_sample_offsets_ = grad_sample_offsets.data<scalar_t>();

        coord_inst2img_gpu_kernel<<<
            GET_BLOCKS(num_kernels),
            CUDA_NUM_THREADS>>>(
            num_kernels,
            data_feature_,
            data_sample_offsets_,
            data_batch_index_,
            num_instance,
            num_points,
            channel_out,
            height_out,
            width_out,
            grad_sample_offsets_,
            grad_output_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in coord_inst2img: %s\n", cudaGetErrorString(err));
  }
}
