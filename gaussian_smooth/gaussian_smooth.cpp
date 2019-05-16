#include <torch/extension.h>
#include "gaussian_smooth.h"


at::Tensor gaussian_smooth_forward(
    at::Tensor input,
    at::Tensor sigma_map,
    int kernel_h, int kernel_w
){
    /**
    * get the input parameter's information
    **/
    int stride_h = 1;
    int stride_w = 1;
    int batch = input.size(0);
    int num_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int sigma_map_channels = sigma_map.size(1);
    int sigma_map_height = sigma_map.size(2);
    int sigma_map_width = sigma_map.size(3);
    int height_out = (input_height - (1 * (kernel_h - 1) + 1)) / stride_h + 1;
    int width_out = (input_width - (1 * (kernel_w - 1) + 1)) / stride_w + 1;
    /**
    * data correctness validation
    **/
    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(sigma_map.type().is_cuda(), "rate_map must be a CUDA tensor");
    AT_ASSERTM(sigma_map_height == height_out, "output height must be same with sigma_map height");
    AT_ASSERTM(sigma_map_width == width_out, "output width must be same with sigma_map width");
    AT_ASSERTM(kernel_h % 2 == 1 || kernel_w % 2 ==1, "kernel_size must be odd number");
    AT_ASSERTM(sigma_map_channels == 1, "sigma_map_channels are not equal to 1.");
    /**
    * derive more information
    **/
    int input_dim = num_channels * input_height * input_width;
    int conv_out_spatial_dim = height_out * width_out;

    /**
    * malloc tmp space and output space
    **/
    auto output = at::empty({batch, num_channels, height_out, width_out}, input.options());
    /**
    * get pointer of the tensors
    **/
    auto input_ptr = input.data<float>();
    auto sigma_map_ptr = sigma_map.data<float>();
    auto output_ptr = output.data<float>();
    
    for (int n = 0; n < batch; ++n) {
        gaussian_smooth_im2col(
            THCState_getCurrentStream(state),
            input_ptr + n * input_dim,
            sigma_map_ptr + n * conv_out_spatial_dim,
            num_channels, input_height, input_width,
            kernel_h, kernel_w,
            stride_h, stride_w,
            height_out, width_out,
            output_ptr + (n * num_channels * conv_out_spatial_dim)
        );
    }
    return output;
}

std::vector<at::Tensor> gaussian_smooth_backward(
    at::Tensor input,
    at::Tensor sigma_map,
    at::Tensor out_grad,
    int kernel_h, int kernel_w
){
    /**
    * get the input parameter's information
    **/
    int stride_h = 1;
    int stride_w = 1;
    int batch = input.size(0);
    int num_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int sigma_map_height = sigma_map.size(2);
    int sigma_map_width = sigma_map.size(3);
    int height_out = (input_height  - (1 * (kernel_h - 1) + 1)) / stride_h + 1;
    int width_out = (input_width - (1 * (kernel_w - 1) + 1)) / stride_w + 1;
    /**
    * data correctness validation
    **/
    AT_ASSERTM(height_out==out_grad.size(2) && width_out == out_grad.size(3),
        "the calculated out shape won't match the out_grad_shape:(%d x %d vs %d x %d)",
            height_out, width_out, out_grad.size(2), out_grad.size(3));
    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(sigma_map.type().is_cuda(), "rate_map must be a CUDA tensor");
    AT_ASSERTM(sigma_map_height == height_out, "output height must be same with sigma_map height");
    AT_ASSERTM(sigma_map_width == width_out, "output width must be same with sigma_map width");
    /**
    * derive more information
    **/
    int input_dim = num_channels * input_height * input_width;
    int conv_out_spatial_dim = height_out * width_out;
    /**
    * malloc tmp space and output space
    **/
    auto grad_input = at::zeros_like(input);
    auto grad_sigma_map = at::zeros_like(sigma_map);
    /**
    * get pointer of the tensors
    **/
    auto input_ptr = input.data<float>();
    auto sigma_map_ptr = sigma_map.data<float>();
    auto out_grad_ptr = out_grad.data<float>();
    
    auto grad_input_ptr = grad_input.data<float>();
    set_zeros(THCState_getCurrentStream(state), batch * num_channels*input_height*input_width, grad_input_ptr);
    auto grad_sigma_map_ptr = grad_sigma_map.data<float>();
    
    for (int n = 0; n < batch; ++n) {
        gaussian_smooth_col2im(
            THCState_getCurrentStream(state),
            out_grad_ptr + n * num_channels * conv_out_spatial_dim,
            input_ptr + n * input_dim,
            sigma_map_ptr + n * conv_out_spatial_dim,
            num_channels, input_height, input_width,
            kernel_h, kernel_w,
            stride_h, stride_w,
            height_out, width_out,
            grad_sigma_map_ptr + n * conv_out_spatial_dim,
            grad_input_ptr + n * input_dim
        );
    }
    return {grad_input, grad_sigma_map};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("forward", &gaussian_smooth_forward, "gaussian_smooth_forward forward (CUDA)");
  m.def("backward", &gaussian_smooth_backward, "gaussian_smooth_backward backward (CUDA)");
}
