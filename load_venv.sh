#!/bin/bash
module --force purge
module load python/3.9

module list

VENV_NAME='homeostatic_dann_2025'
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

export PYTHONPATH=$PYTHONPATH:~/HomeostaticDANN