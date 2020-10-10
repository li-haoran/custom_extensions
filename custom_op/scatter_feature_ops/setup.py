from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='scatter_feature',
    ext_modules=[
        CUDAExtension('scatter_feature_cuda', [
            'src/scatter_feature_cuda.cpp',
            'src/scatter_feature_cuda_kernel.cu',
        ],
        ),#extra_compile_args=['-g']
    ],
    cmdclass={'build_ext': BuildExtension})
