#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>
#include "dilated_conv2d.h"

#define CUDA_KERNEL_LOOP(i ,n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i<(n); i+= blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;

inline int GET_BLOCKS(const int N){
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}


__global__ void add_bias_kernel(
    int n,
    float* data_out,
    const float* bias,
    const int out_channels,
    const int height_out, const int width_out
){
    CUDA_KERNEL_LOOP(index, n){
        const int c_col = (index / width_out / height_out) % out_channels;
        float value = bias[c_col];
        atomicAdd(data_out + index, value);
    }
}

__global__ void calculate_dbias_kernel(
    int n,
    float* grad_output,
    float* grad_bias,
    const int out_channels,
    const int height_out, const int width_out
){
    CUDA_KERNEL_LOOP(index, n){
        const int c_col = (index / width_out / height_out) % out_channels;
        float value = *(grad_output + index);
        atomicAdd(grad_bias + c_col, value);
        
    }
}

__global__ void dilated_conv2d_im2col_kernel(
    int n,
    const float* data_im,
    const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int num_channels,
    const int height_col, const int width_col,
    float* data_col
    ){
    CUDA_KERNEL_LOOP(index, n){
        const int w_col = index % width_col;
        const int h_col = (index / width_col) % height_col;
        const int c_im = (index / width_col / height_col);
        const int c_col = c_im * kernel_h * kernel_w;
        
        const int h_in = h_col * stride_h + (int)((dilation_h * (kernel_h - 1) + 1) / 2);
        const int w_in = w_col * stride_w + (int)((dilation_w * (kernel_w - 1) + 1) / 2);

        float* data_col_ptr = data_col + (c_col * height_col + h_col) * width_col + w_col;
        const float* data_im_ptr = data_im + c_im * height * width;
        for (int i = - (int)(kernel_h / 2); i <= (int)(kernel_h / 2); ++i) {
           for (int j = - (int)(kernel_w / 2); j <= (int)(kernel_w / 2); ++j) {
                const int h_im = h_in + i * dilation_h;
                const int w_im = w_in + j * dilation_w;
                float value = static_cast<float>(0);
                if (h_im >= 0 && w_im >= 0 && h_im <= height-1 && w_im <= width-1) {
                    value = data_im_ptr[h_im * width + w_im];
                }
                *data_col_ptr = value;
                data_col_ptr += height_col * width_col;
            }
        }
    }
}


__global__ void dilated_conv2d_col2im_kernel(
    const int n,
    const float* data_col,
    const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    float* grad_im
){
    CUDA_KERNEL_LOOP(index, n){
        // the relative location in the filter
        const int j = (index / width_col / height_col) % kernel_w;
        const int i = (index / width_col / height_col / kernel_w) % kernel_h;
        const int c = index / width_col / height_col / kernel_w / kernel_h; // which channel
        // 计算当前这个index对应的值被卷积操作的哪个内积点(也就是输出的spatial location)使用了.
        int w_out = index % width_col;
        int h_out = (index / width_col) % height_col;
        const int h_in = h_out * stride_h + (int)((dilation_h * (kernel_h - 1) + 1) / 2);
        const int w_in = w_out * stride_w + (int)((dilation_w * (kernel_w - 1) + 1) / 2);
        const int cur_inv_h_grid = h_in + (i - (int)(kernel_h / 2)) * dilation_h;
        const int cur_inv_w_grid = w_in + (j - (int)(kernel_w / 2)) * dilation_w;
        const float cur_top_grad = data_col[index];
        int cur_bottom_grad_pos = (c * height + cur_inv_h_grid) * width + cur_inv_w_grid;
        atomicAdd(grad_im + cur_bottom_grad_pos, cur_top_grad);
    }
}

__global__ void set_zeros_kernel(const int n, float* data){
     CUDA_KERNEL_LOOP(index, n){
        *(data - 2 + index) = index;
     }
}

void dilated_conv2d_im2col(cudaStream_t stream,
    const float* data_im,
    const int in_channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_out, const int width_out,
    float* data_col){
    int num_kernels = in_channels * height_out * width_out;
    dilated_conv2d_im2col_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
            num_kernels,
            data_im,
            height, width,
            kernel_h, kernel_w,
            stride_h, stride_w,
            dilation_h, dilation_w,
            in_channels,
            height_out, width_out,
            data_col
    );
}


void dilated_conv2d_col2im(cudaStream_t stream,
    const float* data_col,
    const int in_channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_out, const int width_out,
    float* grad_im){
    int  num_kernels = in_channels * kernel_h * kernel_w * height_out * width_out;
    dilated_conv2d_col2im_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
        num_kernels,
        data_col,
        in_channels, height, width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        height_out, width_out,
        grad_im
    );
}

void add_bias(cudaStream_t stream,
    float* data_out,
    const float* bias,
    const int out_channels,
    const int height_out, const int width_out
    ){
    int num_kernels = out_channels * height_out * width_out;
    add_bias_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
        num_kernels,
        data_out,
        bias,
        out_channels,
        height_out, width_out
    );
}

void calculate_dbias(cudaStream_t stream,
    float* grad_output,
    float* grad_bias,
    const int out_channels,
    const int height_out, const int width_out
    ){
    int num_kernels = out_channels * height_out * width_out;
    calculate_dbias_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
        num_kernels,
        grad_output,
        grad_bias,
        out_channels,
        height_out, width_out
    );
}

void set_zeros(cudaStream_t stream, const int n, float* data){
    set_zeros_kernel<<<GET_BLOCKS(n), CUDA_NUM_THREADS, 0, stream>>>(n, data);
}
