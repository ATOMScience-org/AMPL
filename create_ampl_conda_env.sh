#/bin/bash
# 
# A script to create a Conda environment for AMPL development.
#

# 1. create a conda env
start=$SECONDS

conda_name=$1

# if the user doesn't proivde a new conda env name from the command line, prompt the user for one
if [ "$conda_name" = "" ];
then
    echo -n "Enter a new environment for Conda: "
    read conda_name
fi
echo "Will create a new environment called '$conda_name'."

cd conda
conda create -y -n $conda_name --file conda_package_list.txt 

source  /usr/share/miniconda/etc/profile.d/conda.sh
conda activate $conda_name

conda info | grep -i 'base environment'
conda info | grep -i 'active environment'

# 2. activate the new env and run pip install
echo "Done creating the conda env."
echo "Activate the environment '$conda_name'."

echo "Run pip install"
pip install -r pip_requirements.txt

# 3. to fix the error:
# 'tensorflow.python.training.experimental.mixed_precision' has no attribute '_register_wrapper_optimizer_cls'

pip uninstall -y keras
pip install -U tensorflow==2.8.0 keras==2.8.0

duration=$(( SECONDS - start ))
