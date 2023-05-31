## Setup TensorFlow to Run GPU

To get TensorFlow to run GPU requires extra steps to point TF to the required libraries.

> ***Note***:
> *If you don't have it seup properly, you may see these errors:*

```
2023-05-17 14:29:29.886963: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-05-17 14:29:38.750301: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer.so.7'; dlerror: libnvinfer.so.7: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: ~/lib:/usr/tce/packages/cuda/cuda-11.3.0/lib64
2023-05-17 14:29:38.752222: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer_plugin.so.7'; dlerror: libnvinfer_plugin.so.7: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: ~/lib:/usr/tce/packages/cuda/cuda-11.3.0/lib64
2023-05-17 14:29:38.752254: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.
```

### Solution

```
pip install --upgrade nvidia-tensorrt                    # install nvidia-tensorrt if you haven't done so
 
cd $YOUR_ENV/lib/python3.9/site-packages                 # cd to your environment's python site-packages directory

find . -type f -name libnvi*.so.* -print                 # find out where libnvinfer.so* locates. Mine is in tensorrt_libs
cd tensorrt_libs                                         # cd to the directory that has libnvinfer.so*
ln -s libnvinfer.so.8 libnvinfer.so.7                    # TF is looking for .7. link .8 to libnvinfer.so.7
ln -s libnvinfer_plugin.so.8 libnvinfer_plugin.so.7

module load cuda/11.3.0                                  # setup for cuda, use the cuda version that's available

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$YOUR_ENV/lib/python3.9/site-packages/tensorrt:$YOUR_ENV/lib/python3.9/site-packages/tensorrt_libs 
                                                         # define the LD_LIBRARY_PATH to include the ones under tensorrt, tensorrt_libs

which nvcc                                               # find out where cuda library is. 
                                                         # mine is /usr/tce/packages/cuda/cuda-11.3.0/bin/nvcc
ls  -R /usr/tce/packages/cuda/cuda-11.3.0/* |grep libdevice    # find where your libdevice library sits under the cuda library
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/tce/packages/cuda/cuda-11.3.0      # set XLA_FLAGS to your libdevice library 
```
