#!/usr/bin/bash

ml rocm/5.4.3
#ml rocmcc/5.4.3-magic
ml python/3.9.12

if [[ $# -ne 2 ]] ; then
    echo 'Script needs as input two arguments'
    echo 'a venv directory to activate; and one of cluster, scatter, sparse, spline, geometric, or full'
    exit 1
fi

source $1

# we assume torch is installed in venv
torchlib=`./torch_path.py`/lib
echo "torchlib=$torchlib"
# only add to LD_LIBRARY_PATH if not already there
# do something like this in runtime env too
if [[ ! $LD_LIBRARY_PATH =~ "torch/lib" ]]; then
  export LD_LIBRARY_PATH=$torchlib:$LD_LIBRARY_PATH
fi

if [[ $2 =~ .*"cluster".* || $2 == "full" ]]; then
pushd tpl/pyg-rocm-build/pytorch_cluster-1.6.1/
echo "Building torch_cluster"
python3 setup.py build_ext -j 24
python3 setup.py bdist_wheel
python3 -m pip install --no-deps --force-reinstall dist/torch_cluster*.whl
popd
fi

if [[ $2 =~ .*"scatter".* || $2 == "full" ]]; then
pushd tpl/pyg-rocm-build/pytorch_scatter-2.1.1/
echo "Building torch_scatter"
python3 setup.py build_ext -j 24
python3 setup.py bdist_wheel
python3 -m pip install --no-deps --force-reinstall dist/torch_scatter*.whl
popd
fi

if [[ $2 =~ .*"sparse".* || $2 == "full" ]]; then
pushd tpl/pyg-rocm-build/pytorch_sparse-0.6.17/
echo "Building torch_sparse"
python3 setup.py build_ext -j 24
python3 setup.py bdist_wheel
python3 -m pip install --no-deps --force-reinstall dist/torch_sparse*.whl
popd
fi

if [[ $2 =~ .*"spline".* || $2 == "full" ]]; then
pushd tpl/pyg-rocm-build/pytorch_spline_conv-1.2.2/
echo "Building torch_spline_conv"
python3 setup.py build_ext -j 24
python3 setup.py bdist_wheel
python3 -m pip install --no-deps --force-reinstall dist/torch_spline*.whl
popd
fi

if [[ $2 =~ .*"geometric".* || $2 == "full" ]]; then
pushd tpl/pyg-rocm-build/pytorch_geometric-2.3.1/
echo "Building torch_geometric"
python3 setup.py bdist_wheel
python3 -m pip install --no-deps --force-reinstall dist/torch_geometric*.whl
popd
fi

echo "Done Installing pyg!!"



