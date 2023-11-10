#!/usr/bin/env python3

import io
import os
import re

from setuptools import find_packages, setup


# Get version
def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


readme = open("README.md").read()
version = find_version("fande", "__init__.py")

############################# Torch dependency is heavy #########################################
# torch_min = "1.9"
# install_requires = [">=".join(["torch", torch_min]), "scikit-learn", "scipy"]
# # if recent dev version of PyTorch is installed, no need to install stable
# try:
#     import torch

#     if torch.__version__ >= torch_min:
#         install_requires = []
# except ImportError:
#     pass
#################################################################################################
install_requires = ["numpy"]


# Run the setup
setup(
    name="fande",
    version=version,
    description="Molecular Energy and Force fitting with scalable Gaussian Processes",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Mikhail Tsitsvero",
    url="example.com",
    author_email="tsitsvero@gmail.com",
    project_urls={
        "Documentation": "https://fande.readthedocs.io",
        "Source": "https://github.com/chem-gp/fande/",
    },
    license="MIT",
    classifiers=["Development Status :: 0 - Beta", "Programming Language :: Python :: 3"],
    packages=find_packages(exclude=["test", "test.*"]),
    python_requires=">=3.6",
    install_requires=install_requires,
    extras_require={
        "dev": ["black", "twine", "pre-commit"],
        "docs": ["ipython", "ipykernel", "sphinx<3.0.0", "sphinx_rtd_theme", "nbsphinx", "m2r"],
        "examples": ["ipython", "jupyter", "matplotlib", "scipy", "torchvision", "tqdm"],
        "pyro": ["pyro-ppl>=1.0.0"],
        "keops": ["pykeops>=1.1.1"],
        "test": ["flake8==4.0.1", "flake8-print==4.0.0", "pytest", "nbval"],
    },
    test_suite="test",
)