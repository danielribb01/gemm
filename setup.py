from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="neoGEMM",
    ext_modules=[
        CUDAExtension(
            name="neoGEMM",
            sources=["neogemmV02.cu"],
            include_dirs=[
                "/usr/local/lib/python3.12/dist-packages/pybind11/include"  # Caminho para pybind11 headers
            ],
            extra_compile_args={
                "cxx": ["-std=c++17"],
                "nvcc": ["-std=c++17", "-O3", "-DNDEBUG", "-w", "-expt-relaxed-constexpr", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "--expt-extended-lambda", "--use_fast_math", "-Xcompiler=-fPIC ", "-Xcompiler=-Wno-psabi", "-Xcompiler=-fno-strict-aliasing", "-gencode", "arch=compute_90a,code=sm_90a", "-lineinfo"]
            },
            libraries=["cuda", "cublas"],
            define_macros=[("__CUDA_NO_BFLOAT16_CONVERSIONS__", "0")]
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)


