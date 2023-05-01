#!/bin/bash

conda create -n fande -c conda-forge -y python gpytorch pytorch-lightning cudatoolkit=11.3.1 ase qe xtb xtb-python dftbplus plumed dscribe ipykernel matplotlib networkx gcc clang cmake gfortran
# torch import problems with some versions of cudatoolkit
conda activate fande

# building and installing librascal
cd ..
# git clone https://github.com/lab-cosmo/librascal
git clone https://github.com/chem-gp/librascal.git
cd librascal
mkdir build
cd build
cmake ..
make
cd ..
pip install .

# building and installing i-pi
# https://github.com/i-pi/i-pi
cd ..
# git clone https://github.com/i-pi/i-pi.git
git clone https://github.com/chem-gp/i-pi.git
cd drivers/f90
make
cd ../..

# echo 'export APP=123' >> ~/.bashrc #append to bashrc

# Run examples:
# cd examples/tutorial/tutorial-1/
# i-pi tutorial-1.xml > log &
# i-pi-driver -h localhost -p 31415 -m sg -o 15 &
# i-pi-driver -h localhost -p 31415 -m sg -o 15 &
# tail -f log
