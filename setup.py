from setuptools import setup, find_packages

setup(
    name="hwa-analog-training",
    version="0.1.0",
    author="HWA Research",
    description="Hardware-Aware Training for Analog In-Memory Computing",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/hwa-analog-training",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.15",
        "numpy>=1.21",
        "datasets>=2.0",  # For WikiText-2
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="deep-learning, analog-computing, hardware-aware-training, pcm, neuromorphic",
)
