# setup.py for esrgan-upscale repository

from setuptools import setup, find_packages

setup(
    name="esrgan-upscale",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "opencv-python",
        "numpy",
    ],
)
