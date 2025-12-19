from setuptools import setup, find_packages

setup(
    name="polytope",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pycddlib"
    ],
    python_requires='>=3.8',
    description="3D Polytope visualization and H/V-rep utilities",
    author="Steve Tonneau",
    url="https://github.com/ipab-rwa/polytope", 
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)