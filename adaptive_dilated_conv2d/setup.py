from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
setup(name='adaptive_dilated_conv2d',
      ext_modules=[CUDAExtension('adaptive_dilated_conv2d_gpu', ['adaptive_dilated_conv2d.cpp', 'adaptive_dilated_conv2d_cuda.cu']),],
      cmdclass={'build_ext': BuildExtension})
