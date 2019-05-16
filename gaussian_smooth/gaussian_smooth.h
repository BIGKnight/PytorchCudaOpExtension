#ifndef GAUSSIAN_SMOOTH
#define GAUSSIAN_SMOOTH
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
extern THCState *state;
typedef std::vector<int> TShape;

void gaussian_smooth_im2col(cudaStream_t stream,
    const float* data_im, const float* sigma,
    const int num_channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int height_out, const int width_out,
    float* data_col);

void gaussian_smooth_col2im(
    cudaStream_t stream,
    const float* out_grad, const float* data_im, const float* sigma,
    const int num_channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int height_out, const int width_out,
    float* grad_sigma_map, float* grad_im
);
void set_zeros(cudaStream_t stream, const int n, float* data);
#endif
