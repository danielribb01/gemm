from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="neogemm",
    ext_modules=[
        CUDAExtension(
            name="neogemm",
            sources=["neogemm.cu"],
            include_dirs=[
                "/usr/local/lib/python3.12/dist-packages/pybind11/include"  # Caminho para pybind11 headers
            ],
            extra_compile_args={
                "cxx": ["-std=c++17"],
                "nvcc": ["-std=c++17", "-O3"]
            }
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
