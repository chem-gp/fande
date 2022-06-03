<img src="https://user-images.githubusercontent.com/25351170/171554530-6a38f595-27fa-4b97-be30-7a279b17abd2.png" width="50">

# Description
Fitting molecular Energies, Forces and Chemical shifts with scalable Gaussian Processes with SVI 

Project scope is for those doing research in probabilistic machine learning applied to chemical physics problems

Currently fande uses [Gpytorch](https://github.com/cornellius-gp/gpytorch)/[Pytorch](https://github.com/pytorch/pytorch)/[Pyro](https://github.com/pyro-ppl/pyro) for ML modeling and [DScribe](https://github.com/SINGROUP/dscribe) library for computing descriptors

In future we plan to add [GPFlow](https://github.com/GPflow/GPflow)/[TensorFlow](https://github.com/tensorflow/tensorflow)/[TensorflowProbability](https://github.com/tensorflow/probability) backends as well as add support for [librascal](https://github.com/lab-cosmo/librascal) library to compute descriptors


## Examples

Collection with examples is available [here](https://github.com/chem-gp/examples). Sample molecular dynamics trajectories are available [here](https://figshare.com/projects/fande-data/140660).

### Fitting forces and energies

[Fitting energies/forces from molecular dynamics by the GP](https://github.com/chem-gp/examples/blob/main/fande-examples/energy_forces_fit.ipynb).

![image](https://user-images.githubusercontent.com/25351170/171811297-7a9541c5-df9b-4ea6-87c6-79e1180bbe64.png)



### Chemical transformation network discovery aided by GPs

<img src="https://user-images.githubusercontent.com/25351170/171550682-25ea416f-bc54-4373-9b31-1fdbc1f5381e.gif" width="250">

Theory on barrier crossings with biased gradients is described [here](https://arxiv.org/pdf/2202.13011.pdf).

## Currently implemented methods

| Method  | Availability |
| ------------- | ------------- |
| method... | 0  |
| Content Cell  | Content Cell  |


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


