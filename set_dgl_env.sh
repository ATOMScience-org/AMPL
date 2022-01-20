#/usr/bin/bash

# An example script to setup the run environment for DeepChem dgl
# 
# Note: Please change your conda home directory if it's different from the example.

# to run, 
#
# $ source set_dgl_env.sh $test_env 
#
# where $test_env is the name of the Conda environment to run dgl in.

env=$1
module load cuda/11.1
export LD_LIBRARY_PATH=$HOME/.conda/envs/$env/lib:$LD_LIBRARY_PATH # change to your conda/envs path

