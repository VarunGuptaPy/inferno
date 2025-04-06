from setuptools import setup, find_packages

# Import version from package
from inferno import __version__, __author__, __email__

setup(
    name="inferno",
    version=__version__,
    description="A professional, production-ready inference server for running any AI model with universal model compatibility and multi-hardware support",
    long_description="Inferno is a high-performance inference server that can run any AI model from Hugging Face, local files, or GGUF format. It features automatic memory management, hardware detection, and supports CPU, GPU, TPU, and Apple Silicon platforms.",
    long_description_content_type="text/plain",
    author=__author__,
    author_email=__email__,
    url="https://github.com/HelpingAI/inferno",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.22.0",
        "pydantic>=1.10.0",
        "requests>=2.28.0",
        "tqdm>=4.64.0",
        "py-cpuinfo>=9.0.0",
        "bitsandbytes>=0.40.0",
        "psutil>=5.9.0",
        "huggingface_hub>=0.16.0",
        "rich>=10.0.0",
    ],
    extras_require={
        "tpu": ["torch_xla"],
        "gguf": ["llama-cpp-python", "cmake", "ninja"],
    },
    entry_points={
        "console_scripts": [
            "inferno=inferno.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",  # Closest standard classifier
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)
