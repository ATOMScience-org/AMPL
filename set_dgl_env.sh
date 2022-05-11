#/usr/bin/bash

# Some models use DeepChem dgl which requires CUDA. This is an example script to setup the run environment for dgl.
# 
# Note:
#
# 1) This doesn't apply to AMPL Docker image
# 2) Please change your conda home directory if it's different from the example.

# to run, 
#
# $ source set_dgl_env.sh $test_env 
#
# where $test_env is the name of your current Conda environment to run dgl in.

env=$1

if [ "$env" = "" ];
then
    echo -n "Enter the current cond environment: "
    read env
fi

# check if it's running on LC or not
if [ -d "/p/vast1" ]
then
    echo "Run on LC. Load CUDA."
    module load cuda/11.1
else
    echo "Not running on LC. Suggestions: "
    echo "   1) Install CUDA"
    echo "      https://developer.nvidia.com/cuda-11.1.0-download-archive"
    echo "   2) Set up an environment variable to use CPU instead of GPU"
    echo "      $ export CUDA_VISIBLE_DEVICES=''"
fi

# If you are running in a conda environment, add the environment/lib to LD_LIBRARY_PATH
#
# For example: export LD_LIBRARY_PATH=$HOME/.conda/envs/atomsci/lib:$LD_LIBRARY_PATH
#
# where `atomsci` is the conda environment that's running in
export LD_LIBRARY_PATH=$HOME/.conda/envs/$env/lib:$LD_LIBRARY_PATH # change to your conda/envs path

