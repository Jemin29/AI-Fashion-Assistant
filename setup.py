# =============================================================================
# AI-Powered Fashion Design Assistant
# setup.py — Package Installation Configuration
# =============================================================================

from setuptools import setup, find_packages

setup(
    name="fashion-ai-assistant",
    version="1.0.0",
    description="AI-Powered Fashion Design Assistant",
    author="Fashion AI Team",
    python_requires=">=3.10",
    package_dir={"": "."},
    packages=find_packages(
        exclude=["tests", "tests.*", "notebooks", "docs"]
    ),
    install_requires=[
        "python-dotenv>=1.0.1",
        "pyyaml>=6.0.1",
        "loguru>=0.7.2",
        "tqdm>=4.66.2",
        "numpy>=1.26.4",
        "Pillow>=10.3.0",
        "pandas>=2.2.1",
        "pyarrow>=15.0.2",
        "pydantic>=2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.2.0",
            "pytest-asyncio>=0.23.6",
            "pytest-cov>=5.0.0",
            "black>=24.4.2",
            "isort>=5.13.2",
            "flake8>=7.0.0",
            "mypy>=1.10.0",
        ],
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
