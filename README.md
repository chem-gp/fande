<img src="https://user-images.githubusercontent.com/25351170/171819527-8718a6ae-db35-48f8-9364-9cc11cc0fb8b.png" width="100">


# Description
Fitting molecular Energies, Forces and Chemical shifts with scalable Gaussian Processes trained by [stochastic variational inference](https://jmlr.org/papers/volume14/hoffman13a/hoffman13a.pdf) on multiple GPUs.

Project scope is for those doing research in probabilistic machine learning applied to chemical physics problems

Currently fande uses [GPyTorch](https://github.com/cornellius-gp/gpytorch)/[PyTorch](https://github.com/pytorch/pytorch)/[Pyro](https://github.com/pyro-ppl/pyro) for ML modeling and [librascal](https://github.com/lab-cosmo/librascal) library for computing invariants.



## Examples

...

<!-- Collection with examples is available [here](https://github.com/chem-gp/examples). Sample molecular dynamics trajectories are available [here](https://figshare.com/projects/fande-data/140660). -->

<!-- ### Fitting forces and energies -->

<!-- [Fitting energies/forces from molecular dynamics by the GP](https://github.com/chem-gp/examples/blob/main/fande-examples/energy_forces_fit.ipynb). -->


<!-- <img src="https://user-images.githubusercontent.com/25351170/171815059-1ce8ad74-e7fd-4c89-b75b-6ebe4ec2ccdc.gif" width="250">

Example force fits (Fx,Fy,Fz) with uncertainty estimates for MD continuation:

![image](https://user-images.githubusercontent.com/25351170/171811297-7a9541c5-df9b-4ea6-87c6-79e1180bbe64.png) -->


<!-- ### Chemical transformation network discovery -->

<!-- [Crossing barriers with fande.explore](https://github.com/chem-gp/examples/blob/main/fande-examples/fande_explore_crossing_barriers.ipynb) -->

<!-- <img src="https://user-images.githubusercontent.com/25351170/171550682-25ea416f-bc54-4373-9b31-1fdbc1f5381e.gif" width="250">
<img src="https://user-images.githubusercontent.com/25351170/172145099-3fcd9649-1e08-4f49-8544-f5de05ebccd2.png" width="250"> -->


<!-- Example with ASE `atoms`:
```python
from ase.build import molecule
from fande.explore import ForcedExplorer

atoms = molecule("C2H6CHOH")
fx = ForcedExplorer(atoms, logfile='data/dump/explore_log.log')
traj, e_path, atoms_opt, energy_opt, energy_pre_opt = fx.single_forced_run(atoms, [[4] , [5] ], force=9.0 )
```

Theory on barrier crossings is similar to the one [given here](https://arxiv.org/pdf/2202.13011.pdf). -->

## Currently implemented methods

<!-- | Method  | Availability |
| ------------- | ------------- |
| method... | 0  |
| Content Cell  | Content Cell  | -->


## Install

<!-- Currently we provide install through [conda]()

1. Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)

2. Clone repository 
``` bash
git clone https://github.com/chem-gp/fande
cd fande
git clone https://github.com/chem-gp/examples
```
3. Create conda environment
```
conda env create -f fande.yml
```
4. Activate environment
```
conda activate fande
```
5. Install package
```
pip install -e ../fande
```
7. All should work, you can run [examples]([https://github.com/chem-gp/examples](https://github.com/chem-gp/examples/tree/main/fande-examples))! -->



Install by pip:
```
pip install git+https://github.com/chem-gp/fande
pip install git+https://github.com/chem-gp/fande --force-reinstall
```