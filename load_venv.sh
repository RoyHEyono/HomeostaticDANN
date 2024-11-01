#!/bin/bash
module purge
module load python/3.7 
module load cuda/11.1/cudnn/8.0
module load python/3.7/cuda/11.1/cudnn/8.0/pytorch/1.8.1

module list

VENV_NAME='danns_eg'
VENV_DIR=$HOME'/venvs/'$VENV_NAME

echo 'Loading virtual env: '$VENV_NAME' in '$VENV_DIR

# Activate virtual enviroment if available

# Remeber the spacing inside the [ and ] brackets! 
if [ -d $VENV_DIR ]; then
	echo "Activating danns_venv"
    source $VENV_DIR'/bin/activate'
else 
	echo "ERROR: Virtual enviroment does not exist... exiting"
fi 

export PYTHONPATH=$PYTHONPATH:~/dann_rnns