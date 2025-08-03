# modified from https://github.com/jcornford/dann_rnns/blob/main/make_venv.sh
#
module --force purge
module load python/3.9


VENV_NAME='.venv'
VENV_DIR=$HOME'/HomeostaticDANN/'$VENV_NAME

echo 'Building virtual env: '$VENV_NAME' in '$VENV_DIR

mkdir $VENV_DIR
# Add .gitignore to folder just in case it is in a repo
# Ignore everything in the directory apart from the gitignore file
echo "*" > $VENV_DIR/.gitignore
echo "!.gitignore" >> $VENV_DIR/.gitignore

virtualenv $VENV_DIR

source $VENV_DIR'/bin/activate'
pip install --upgrade pip
pip cache purge

# install python packages not provided by modules

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir
pip install pandas==1.5.3


pip install matplotlib==3.5.3
pip install scipy==1.12.0
pip install numpy==1.22.4
pip install scikit-learn==1.4.0

# install python packages not provided by modules
pip install matplotlib pandas scipy numpy #wandb
pip install pillow
pip install wandb
pip install hydra-core --upgrade
pip install tqdm

pip install ipython --ignore-installed
pip install ipykernel

# Orion installations
pip install orion 

# set up MILA jupyterlab
echo which ipython
ipython kernel install --user --name=homeostatic_dann_2025_kernel

pip3 install -e .




