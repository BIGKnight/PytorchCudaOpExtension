import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import Module
import adaptive_dilated_conv2d_gpu as adaptive_dilated_conv2d


class AdaptiveDilatedConv2dFunction(Function):
    @staticmethod
    def forward(ctx, *args):
        if len(args) != 6:
            print("wrong input parameters number, check the input")
            return
        input = args[0]
        weights = args[1]
        rate_map = args[2]
        bias = args[3]
        ctx.stride_h = args[4]
        ctx.stride_w = args[5]
        output = adaptive_dilated_conv2d.forward(input, weights, rate_map, bias, ctx.stride_h, ctx.stride_w)
        ctx.save_for_backward(input, weights, rate_map, bias)
        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        if len(grad_outputs) != 1:
            print("Wrong output number, check your output")
            return
        input, weights, rate_map, bias = ctx.saved_tensors
        grad = grad_outputs[0].clone()
        grad_input, grad_weight, grad_rate_map, grad_bias = adaptive_dilated_conv2d.backward(input, weights, rate_map, bias, grad, ctx.stride_h, ctx.stride_w)
        return grad_input, grad_weight, grad_rate_map, grad_bias, None, None


class AdaptiveDilatedConv2dLayer(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride_h, stride_w, img_scale, dilation):
        super(AdaptiveDilatedConv2dLayer, self).__init__()
        self.kernel_size = kernel_size
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.float32))
        self.rates = nn.Parameter(torch.ones(1, 1, img_scale, img_scale,  dtype=torch.float32) * dilation)
        self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float32))
        nn.init.xavier_uniform_(self.weight, gain=1)
        
    def forward(self, inputs):
        h_in, w_in = inputs.shape[2:4]
        h_out = (h_in - (1 * (self.kernel_size - 1) + 1)) // self.stride_h + 1 # already add padding in former op, thus it do not worry about the division
        w_out = (w_in - (1 * (self.kernel_size - 1) + 1)) // self.stride_w + 1
        rate_map = nn.functional.interpolate(self.rates, size=[h_out, w_out], mode='bilinear', align_corners=False)
        return AdaptiveDilatedConv2dFunction.apply(inputs, self.weight, rate_map, self.bias, self.stride_h, self.stride_w)

    
class BasicAdaptiveDilatedConv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, scale=100, dilation=1):
        super(BasicAdaptiveDilatedConv2D, self).__init__()
        self.stride = stride
        self.pad = (kernel_size // 2)
        self.adaptive_dilated_conv2d = AdaptiveDilatedConv2dLayer(in_channels, out_channels, kernel_size, self.stride, self.stride, scale, dilation)
        
    def forward(self, x):
        x = torch.nn.functional.pad(x, [self.pad, self.pad, self.pad, self.pad ])
        return self.adaptive_dilated_conv2d(x)

