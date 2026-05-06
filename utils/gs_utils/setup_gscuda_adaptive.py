from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os


file_path = "utils/gs_cuda_dmax_adaptive" 

setup(
    name="gscuda_adaptive", 
    ext_modules=[
        CUDAExtension(
            name="gscuda_adaptive", 
            sources=[
                os.path.join(file_path, "gswrapper_adaptive.cpp"),
                os.path.join(file_path, "gs_adaptive.cu")
            ],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)