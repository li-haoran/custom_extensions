from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='points_collect',
    ext_modules=[
        CUDAExtension('points_collect_cuda', [
            'src/points_collect_cuda.cpp',
            'src/points_collect_cuda_kernel.cu',
        ],
        ),#extra_compile_args=['-g']
    ],
    cmdclass={'build_ext': BuildExtension})
