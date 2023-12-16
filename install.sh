#!/bin/bash

# conda create -n fande -c conda-forge -y python gpytorch pytorch-lightning cudatoolkit=11.3.1 ase qe xtb xtb-python dftbplus plumed py-plumed dscribe ipykernel matplotlib networkx clang cmake gfortran
# torch import problems with some versions of cudatoolkit

conda create -n env -c conda-forge python=3.8 cmake clang pytorch gpytorch numpy scipy ase

conda activate fande
#to install librascal DO NOT INSTALL GCC WITHIN CONDA ENVIRONMENT! Aslo build does not work with Python 3.11, 3.12

pip install git+https://github.com/lab-cosmo/librascal

pip install git+https://github.com/chem-gp/fande


# building and installing i-pi
git clone https://github.com/chem-gp/i-pi.git
cd drivers/f90
make
cd ../..

