from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
setup(name='gaussian_smooth',
      ext_modules=[CUDAExtension('gaussian_perspective_smooth', ['gaussian_smooth.cpp', 'gaussian_smooth_cuda.cu']),],
      cmdclass={'build_ext': BuildExtension})
