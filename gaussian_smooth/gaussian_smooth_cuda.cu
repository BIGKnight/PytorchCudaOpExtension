#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>
#include "gaussian_smooth.h"

#define CUDA_KERNEL_LOOP(i ,n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i<(n); i+= blockDim.x * gridDim.x)

#define SQRT_2Pi 2.5066282746310002
#define Pi 3.1415926

const int CUDA_NUM_THREADS = 1024;

inline int GET_BLOCKS(const int N){
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

__device__ float gaussian_weights_gen(
    const float sigma,
    const int center_i, const int center_j,
    const int k, const int l
){
//     float coefficient = 1 / (2 * Pi * sigma * sigma);
    float L2_distance = (k - center_i) * (k - center_i) + (l - center_j) * (l - center_j);
    float exp_item = exp(-(L2_distance) / (2 * sigma * sigma));
//     return coefficient * exp_item;
    return exp_item;
}

__device__ float gaussian_derivative_gen(
    const float sigma,
    const int center_i, const int center_j,
    const int k, const int l
){
    float L2_distance = (k - center_i) * (k - center_i) + (l - center_j) * (l - center_j);
    float part_1 = exp(-(L2_distance) / (2 * sigma * sigma));
//     float part_2 = L2_distance / (Pi * 2 * sigma * sigma * sigma * sigma * sigma);
//     float part_3 = 1 / (Pi * sigma * sigma * sigma);
    float part_2 = L2_distance / (sigma * sigma * sigma);
//     float part_3 = 2 / sigma;
//     return part_1 * (part_2 - part_3);
    return part_1 * part_2;
}

__global__ void gaussian_smooth_im2col_kernel(
    int n,
    const float* data_im,
    const float* sigma_map,
    const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int num_channels,
    const int height_col, const int width_col,
    float* output
    ){
    CUDA_KERNEL_LOOP(index, n){
        const int w_col = index % width_col;
        const int h_col = (index / width_col) % height_col;
        const int c_im = (index / width_col / height_col) % num_channels;
        const float sigma = sigma_map[h_col * width_col + w_col];
        
        const int h_in = h_col * stride_h + (int)((kernel_h - 1 ) / 2);
        const int w_in = w_col * stride_w + (int)((kernel_w - 1 ) / 2);

        float* output_ptr = output + (c_im * height_col + h_col) * width_col + w_col;
        const float* data_im_ptr = data_im + c_im * height * width;
        float collective_val = static_cast<float>(0);
        float collective_weight = static_cast<float>(0);
        
        for (int i = - (int)(kernel_h / 2); i <= (int)(kernel_h / 2); ++i) {
            for (int j = - (int)(kernel_w / 2); j <= (int)(kernel_w / 2); ++j) {
                float val = static_cast<float>(0);
                float weight = static_cast<float>(0);
                const int h_im = h_in + i;
                const int w_im = w_in + j;
                if (h_im >= 0 && w_im >= 0 && h_im <= height - 1 && w_im <= width - 1) {
                    val = data_im_ptr[h_im * width + w_im];
                    weight = gaussian_weights_gen(sigma, h_in, w_in, h_im, w_im);
                    collective_val += val * weight;
                    collective_weight += weight;
                }
            }
        }
        *output_ptr = collective_val / collective_weight;
    }
}

__global__ void gaussian_smooth_col2im_kernel(
    const int n,
    const float* out_grad,
    const float* data_im,
    const float* sigma_map,
    const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int num_channels,
    const int height_col, const int width_col,
    float* grad_sigma_map, float* grad_im
    ){
    CUDA_KERNEL_LOOP(index, n){
        const int w_col = index % width_col;
        const int h_col = (index / width_col) % height_col;
        const int c_im = (index / width_col / height_col) % num_channels;
        const int cur_index = h_col * width_col + w_col;
        const float sigma = sigma_map[cur_index];
        
        const int h_in = h_col * stride_h + (int)((kernel_h - 1 ) / 2);
        const int w_in = w_col * stride_w + (int)((kernel_w - 1 ) / 2);
        
        const int output_ptr_pos = (c_im * height_col + h_col) * width_col + w_col;
        const float* data_im_ptr = data_im + c_im * height * width;
        float* grad_im_ptr = grad_im + c_im * height * width;
        const float out_grad_cur = out_grad[output_ptr_pos];
        float sum_gaussian = static_cast<float>(0);
        float sum_dgaussian = static_cast<float>(0);
        float sum_x_gaussian = static_cast<float>(0);
        float sum_x_dgaussian = static_cast<float>(0);
        
        for (int i = - (int)(kernel_h / 2); i <= (int)(kernel_h / 2); ++i) {
            for (int j = - (int)(kernel_w / 2); j <= (int)(kernel_w / 2); ++j) {
                float val = static_cast<float>(0);
                float dweight = static_cast<float>(0);
                float weight = static_cast<float>(0);
                const int h_im = h_in + i;
                const int w_im = w_in + j;
                if (h_im >= 0 && w_im >= 0 && h_im <= height - 1 && w_im <= width - 1) {
                    val = data_im_ptr[h_im * width + w_im];
                    dweight = gaussian_derivative_gen(sigma, h_in, w_in, h_im, w_im);
                    weight = gaussian_weights_gen(sigma, h_in, w_in, h_im, w_im);
                    sum_gaussian += weight;
                    sum_dgaussian += dweight;
                    sum_x_gaussian += val * weight;
                    sum_x_dgaussian += val * dweight;
                }
            }
        }
        float sigma_gradient = (sum_x_dgaussian * sum_gaussian - sum_x_gaussian * sum_dgaussian) / (sum_gaussian * sum_gaussian);
        atomicAdd(grad_sigma_map + cur_index, sigma_gradient);
        // grad_im
        for (int i = - (int)(kernel_h / 2); i <= (int)(kernel_h / 2); ++i) {
            for (int j = - (int)(kernel_w / 2); j <= (int)(kernel_w / 2); ++j) {
                float weight = static_cast<float>(0);
                const int h_im = h_in + i;
                const int w_im = w_in + j;
                if (h_im >= 0 && w_im >= 0 && h_im <= height - 1 && w_im <= width - 1) {
                    weight = gaussian_weights_gen(sigma, h_in, w_in, h_im, w_im);
                    atomicAdd(grad_im_ptr + h_im * width + w_im, out_grad_cur * weight / sum_gaussian);
                }
            }
        }
    }
}

__global__ void set_zeros_kernel(const int n, float* data){
    CUDA_KERNEL_LOOP(index, n){
        *data = 0;
    }
}

void gaussian_smooth_im2col(cudaStream_t stream,
    const float* data_im, const float* sigma,
    const int num_channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int height_out, const int width_out,
    float* data_col){
    int num_kernels = num_channels * height_out * width_out;
    gaussian_smooth_im2col_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
            num_kernels,
            data_im,
            sigma,
            height, width,
            kernel_h, kernel_w,
            stride_h, stride_w,
            num_channels, height_out, width_out,
            data_col
    );
}

void gaussian_smooth_col2im(
    cudaStream_t stream,
    const float* out_grad, const float* data_im, const float* sigma,
    const int num_channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int height_out, const int width_out,
    float* grad_sigma_map, float* grad_im
){
    int num_kernels = num_channels * height_out * width_out;
    gaussian_smooth_col2im_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
            num_kernels,
            out_grad, data_im, sigma,
            height, width,
            kernel_h, kernel_w,
            stride_h, stride_w,
            num_channels, height_out, width_out,
            grad_sigma_map, grad_im
    );
}

void set_zeros(cudaStream_t stream, const int n, float* data){
    set_zeros_kernel<<<GET_BLOCKS(n), CUDA_NUM_THREADS, 0, stream>>>(n, data);
}
