#!/usr/bin/env bash

CUSTOM_MODULE_PATH=(/home/zzn/anaconda3/envs/pytorch/lib/python3.6/site-packages/adaptive_dilated_conv2d-0.0.0-py3.6-linux-x86_64.egg)
DYNAMIC_LIB_NAME=(adaptive_dilated_conv2d_gpu.cpython-36m-x86_64-linux-gnu.so)
CURRENT_NEW_DYNAMIC_LIB=(build/lib.linux-x86_64-3.6/adaptive_dilated_conv2d_gpu.cpython-36m-x86_64-linux-gnu.so)

python setup.py install
gcc -pthread -B /home/zzn/anaconda3/envs/pytorch/compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I/home/zzn/anaconda3/envs/pytorch/lib/python3.6/site-packages/torch/lib/include -I/home/zzn/anaconda3/envs/pytorch/lib/python3.6/site-packages/torch/lib/include/torch/csrc/api/include -I/home/zzn/anaconda3/envs/pytorch/lib/python3.6/site-packages/torch/lib/include/TH -I/home/zzn/anaconda3/envs/pytorch/lib/python3.6/site-packages/torch/lib/include/THC -I/usr/local/cuda/include -I/home/zzn/anaconda3/envs/pytorch/include/python3.6m -c adaptive_dilated_conv2d.cpp -o build/temp.linux-x86_64-3.6/adaptive_dilated_conv2d.o -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=adaptive_dilated_conv2d_gpu -D_GLIBCXX_USE_CXX11_ABI=1 -std=c++11

/usr/local/cuda/bin/nvcc -I/home/zzn/anaconda3/envs/pytorch/lib/python3.6/site-packages/torch/lib/include -I/home/zzn/anaconda3/envs/pytorch/lib/python3.6/site-packages/torch/lib/include/torch/csrc/api/include -I/home/zzn/anaconda3/envs/pytorch/lib/python3.6/site-packages/torch/lib/include/TH -I/home/zzn/anaconda3/envs/pytorch/lib/python3.6/site-packages/torch/lib/include/THC -I/usr/local/cuda/include -I/home/zzn/anaconda3/envs/pytorch/include/python3.6m -c adaptive_dilated_conv2d_cuda.cu -o build/temp.linux-x86_64-3.6/adaptive_dilated_conv2d_cuda.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --compiler-options '-fPIC' -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=adaptive_dilated_conv2d_gpu -D_GLIBCXX_USE_CXX11_ABI=1 -std=c++11

g++ -pthread -shared -B /home/zzn/anaconda3/envs/pytorch/compiler_compat -L/home/zzn/anaconda3/envs/pytorch/lib -Wl,-rpath=/home/zzn/anaconda3/envs/pytorch/lib -Wl,--no-as-needed -Wl,--sysroot=/ build/temp.linux-x86_64-3.6/adaptive_dilated_conv2d.o build/temp.linux-x86_64-3.6/adaptive_dilated_conv2d_cuda.o -L/usr/local/cuda/lib64 -lcudart -o build/lib.linux-x86_64-3.6/adaptive_dilated_conv2d_gpu.cpython-36m-x86_64-linux-gnu.so
#rm -f ${CUSTOM_MODULE_PATH}/${DYNAMIC_LIB_NAME}
#mv ${CURRENT_NEW_DYNAMIC_LIB} ${CUSTOM_MODULE_PATH}/
cp -f ${CURRENT_NEW_DYNAMIC_LIB} ${CUSTOM_MODULE_PATH}
