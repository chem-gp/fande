#!/bin/bash

conda create -n fande -c conda-forge -y python gpytorch pytorch-lightning cudatoolkit=11.3.1 ase qe xtb xtb-python dftbplus plumed dscribe ipykernel matplotlib networkx gcc clang cmake
# torch import problems with some versions of cudatoolkit
conda activate fande

# building and installing librascal
git clone https://github.com/lab-cosmo/librascal
cd librascal
mkdir build
cd build
cmake ..
make
cd ..
pip install .
