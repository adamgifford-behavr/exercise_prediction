"""Build python package from src using setuptools"""
from setuptools import find_packages, setup

setup(
    name="src",
    packages=find_packages(),
    version="0.1.0",
    description="Using wearable sensor data to predict type of exercise",
    author="Adam Gifford",
    license="MIT",
)
