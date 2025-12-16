"""Setup script for AMP prediction package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="amp-prediction",
    version="1.0.0",
    author="AMP Research Team",
    author_email="prm@outlook.in",
    description="Enhanced Antimicrobial Peptide Prediction using ESM-650M Embeddings and Ensemble Deep Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PawanRamaMali/amp-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
        "optimization": [
            "optuna>=3.0.0",
        ],
        "tracking": [
            "wandb>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "amp-predict=src.cli:main",
            "amp-train=scripts.train:main",
            "amp-evaluate=scripts.evaluate:main",
            "amp-embed=scripts.generate_embeddings:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml", "*.md"],
    },
)