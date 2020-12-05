from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='unseen_clustering',
    ext_modules=[
        CUDAExtension(
            name='unseen_clustering_cuda', 
            sources = [
            'unseen_clustering_layers.cpp',
            'hard_label_kernel.cu'],
            include_dirs = ['/usr/local/include/eigen3', '/usr/local/include'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
