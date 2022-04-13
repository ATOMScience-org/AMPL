#/bin/bash
# 
# A script to create a Conda environment for AMPL development.
#

# 1. create a conda env
start=$SECONDS
echo -n "Enter a new environment for Conda: "

read conda_name
echo "Will create a new environment called '$conda_name'."

cd conda
conda create -y -n $conda_name --file conda_package_list.txt  

# 2. activate the new env and run pip install
echo "Done creating the conda env."
echo "Activate the environment '$conda_name'."

conda activate $conda_name
echo "Run pip install"
pip install -r pip_requirements.txt

# 3. to fix the error:
# 'tensorflow.python.training.experimental.mixed_precision' has no attribute '_register_wrapper_optimizer_cls'

pip uninstall -y keras
pip install -U tensorflow keras

duration=$(( SECONDS - start ))

echo "All done."
echo "The total execution time: $duration (s)"

