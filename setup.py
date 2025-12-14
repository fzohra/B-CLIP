from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="entmax15_cuda",
    ext_modules=[
        CUDAExtension(
            name="entmax15_cuda",
            sources=["entmax15_cuda_extension.cu"],
            # you can add extra flags here if needed, e.g.:
            # extra_compile_args={"nvcc": ["-O3", "-arch=sm_70"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension}
)
