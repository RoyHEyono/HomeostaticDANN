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

```
python train.py --model.is_dann=True --model.normtype=ln --exp.wandb_project=Normalization_DANN --exp.wandb_entity=project_danns --exp.use_wandb=True --train.use_testset=True
```

### Using conda 
To install using conda, follow the following steps on mila-cluster:

```
. /etc/profile   # could be unnecessary
module load anaconda/3
conda create -y -n ffcv_eg python=3.9 cupy pkg-config compilers libjpeg-turbo opencv pytorch=1.10.2 torchvision=0.11.3 cudatoolkit=11.3 numba -c pytorch -c conda-forge
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

Feb 10th
---------
Results dir for this repo will be in the linclab_users group. Added to the config file
```
project_results_dir: "/network/projects/_groups/linclab_users/danns/subprojects/eg"
```
Mount the directory if desired with something like:
```
sshfs cornforj@login.server.mila.quebec:/network/projects/_groups/linclab_users/danns/subprojects/ ~/mnt/mila -p 2222
```
