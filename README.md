# Description
Fitting molecular Energies, Forces and Chemical shifts with scalable Gaussian Processes with SVI 

Project scope is for those doing research in probabilistic machine learning applied to chemical physics problems

Currently fande uses [Gpytorch](https://github.com/cornellius-gp/gpytorch)/[Pytorch](https://github.com/pytorch/pytorch)/[Pyro](https://github.com/pyro-ppl/pyro) for ML modeling and [DScribe](https://github.com/SINGROUP/dscribe) library for computing descriptors

In future we plan to add [GPFlow](https://github.com/GPflow/GPflow)/[TensorFlow](https://github.com/tensorflow/tensorflow)/[TensorflowProbability](https://github.com/tensorflow/probability) backends as well as add support for [librascal](https://github.com/lab-cosmo/librascal) library to compute spherical invariants


## Examples

Collection with examples is available [here](https://github.com/chem-gp/examples).

### Fitting forces and energies

Example image and code

```
code sample
```

### Chemical transformation network discovery

<img src="https://user-images.githubusercontent.com/25351170/171550682-25ea416f-bc54-4373-9b31-1fdbc1f5381e.gif" width="250">

## Install

Currently we provide install through [conda]() and for Linux distributions

1. Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)

2. Clone repository 
``` bash
git clone https://github.com/chem-gp/fande
cd fande
```
3. Create conda environment
```
conda env create -f conda-fande.yml
```
4. Activate environment
```
conda activate conda-fande
```
5. Install package
```
pip install -e ../fande
```
7. All should work!


