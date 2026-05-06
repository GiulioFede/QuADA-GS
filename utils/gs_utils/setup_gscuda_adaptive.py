from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Cambia questo percorso se hai messo i nuovi file in una cartella diversa
# Es: "utils/gs_cuda_adaptive"
file_path = "utils/gs_cuda_dmax_adaptive" 

setup(
    name="gscuda_adaptive", # <--- Cambiamo nome per non sovrascrivere il vecchio
    ext_modules=[
        CUDAExtension(
            name="gscuda_adaptive", # <--- Questo è il nome che userai in 'import gscuda_adaptive'
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