# danns_eg

Python envs: 
- we want ffcv
- a local python venv 
- a mila python venv

Feb 8th
---------
To allow scripts to import lib modules from anywhere, we install the danns_eg folder as an editable pip package with the following command at repo root:

```
pip install -r requirements.txt -e .
```

To install using conda, follow the following steps on mila-cluster:

```
. /etc/profile   # could be unnecessary
module load anaconda/3
conda create -y -n ffcv_eg python=3.9 cupy pkg-config compilers libjpeg-turbo opencv pytorch torchvision cudatoolkit=11.3 numba -c pytorch -c conda-forge
conda activate ffcv_eg
pip install ffcv
pip install -r requirements.txt -e .
```

To activate the conda environment:
```
. /etc/profile   # could be unnecessary
module load anaconda/3
conda activate ffcv_eg
```
