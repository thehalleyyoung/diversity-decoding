from setuptools import setup, find_packages

setup(
    name="diversity-decoding-arena",
    version="0.1.0",
    description="Diversity metric taxonomy and decoding algorithm evaluation for text generation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Diversity Decoding Arena Contributors",
    python_requires=">=3.9",
    packages=find_packages(where=".", include=["src", "src.*"]),
    package_dir={"": "."},
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "neural": [
            "transformers>=4.20.0",
            "torch>=1.9.0",
            "sentence-transformers>=2.2.0",
        ],
        "openai": [
            "openai>=1.0.0",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
        ],
    },
    entry_points={
        "console_scripts": [
            "diversity-taxonomy=diversity_taxonomy:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
