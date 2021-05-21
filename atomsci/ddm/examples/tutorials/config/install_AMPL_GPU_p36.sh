wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /content/AMPL

cat << "EOF" > AMPL.txt
# This file may be used to create an environment using:
# $ conda create --name <env> --file <this file>
# platform: linux-64
@EXPLICIT
https://conda.anaconda.org/conda-forge/linux-64/_libgcc_mutex-0.1-conda_forge.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/ca-certificates-2020.4.5.1-hecc5488_0.tar.bz2
https://repo.anaconda.com/pkgs/main/linux-64/cudatoolkit-9.0-h13b8566_0.conda
https://conda.anaconda.org/omnia/linux-64/fftw3f-3.3.4-2.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/libgfortran-3.0.0-1.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/libgfortran-ng-7.5.0-hdf63c60_6.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/libstdcxx-ng-9.2.0-hdf63c60_2.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/pandoc-2.9.2.1-0.tar.bz2
https://repo.anaconda.com/pkgs/main/linux-64/cudnn-7.6.5-cuda9.0_0.conda
https://repo.anaconda.com/pkgs/main/linux-64/cupti-9.0.176-0.conda
https://conda.anaconda.org/conda-forge/linux-64/libgomp-9.2.0-h24d8f2e_2.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/openblas-0.2.20-8.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/_openmp_mutex-4.5-0_gnu.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/blas-1.1-openblas.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/libgcc-ng-9.2.0-h24d8f2e_2.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/blosc-1.18.1-he1b5a44_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/bzip2-1.0.8-h516909a_2.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/c-ares-1.15.0-h516909a_1001.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/expat-2.2.9-he1b5a44_2.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/icu-58.2-hf484d3e_1000.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/jpeg-9c-h14c3975_1001.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/libffi-3.2.1-he1b5a44_1007.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/libiconv-1.15-h516909a_1006.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/libllvm8-8.0.1-hc9558a2_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/libsodium-1.0.17-h516909a_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/libuuid-2.32.1-h14c3975_1000.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/lzo-2.10-h14c3975_1000.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/ncurses-6.1-hf484d3e_1002.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/openssl-1.0.2u-h516909a_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/pcre-8.44-he1b5a44_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/pixman-0.34.0-h14c3975_1003.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/pthread-stubs-0.4-h14c3975_1001.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/tbb-2020.1-hc9558a2_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/xorg-kbproto-1.0.7-h14c3975_1002.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/xorg-libice-1.0.10-h516909a_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/xorg-libxau-1.0.9-h14c3975_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/xorg-libxdmcp-1.1.3-h516909a_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/xorg-renderproto-0.11.1-h14c3975_1002.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/xorg-xextproto-7.3.0-h14c3975_1002.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/xorg-xproto-7.0.31-h14c3975_1007.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/xz-5.2.5-h516909a_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/yaml-0.2.4-h516909a_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/zlib-1.2.11-h516909a_1006.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/dbus-1.13.0-h4e0c4b3_1000.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/gettext-0.19.8.1-hc5be6a0_1002.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/hdf5-1.10.6-nompi_h3c11f04_100.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/libpng-1.6.37-hed695b0_1.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/libprotobuf-3.12.1-h8b12597_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/libtiff-4.0.9-h648cc4a_1002.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/libxcb-1.13-h14c3975_1002.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/libxml2-2.9.9-h13577e0_2.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/readline-7.0-hf8c457e_1001.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/tk-8.6.10-hed695b0_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/xorg-libsm-1.2.3-h84519dc_1000.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/zeromq-4.3.2-he1b5a44_2.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/freetype-2.8.1-hfa320df_1.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/glib-2.55.0-h464dc38_2.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/sqlite-3.28.0-h8b20d00_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/xorg-libx11-1.6.9-h516909a_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/fontconfig-2.13.0-hd36ec8e_5.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/gstreamer-1.12.5-h61a6719_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/python-3.6.7-hd21baee_1002.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/xorg-libxext-1.3.4-h516909a_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/xorg-libxrender-0.9.10-h516909a_1002.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/astor-0.7.1-py_0.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/attrs-19.3.0-py_0.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/backcall-0.1.0-py_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/cairo-1.14.12-he56eebe_3.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/decorator-4.4.2-py_0.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/defusedxml-0.6.0-py_0.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/gast-0.3.3-py_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/gst-plugins-base-1.12.5-h3865690_1000.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/idna-2.6-py36_1.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/ipython_genutils-0.2.0-py_1.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/joblib-0.11-py36_0.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/jsonref-0.2-py_0.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/monotonic-1.5-py_0.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/more-itertools-8.3.0-py_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/msgpack-python-0.5.5-py36_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/numpy-1.15.2-py36_blas_openblashd3ea46f_1.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/olefile-0.46-py_0.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/pandocfilters-1.4.2-py_1.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/parso-0.7.0-pyh9f0ad1d_0.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/prometheus_client-0.8.0-pyh9f0ad1d_0.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/ptyprocess-0.6.0-py_1001.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/py-1.8.1-py_0.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/pycparser-2.20-py_0.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/pyparsing-2.4.7-pyh9f0ad1d_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/python_abi-3.6-1_cp36m.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/pytz-2020.1-pyh9f0ad1d_0.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/qtpy-1.9.0-py_0.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/send2trash-1.5.0-py_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/sip-4.18.1-py36hf484d3e_1000.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/six-1.15.0-pyh9f0ad1d_0.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/termcolor-1.1.0-py_2.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/testpath-0.4.4-py_0.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/tqdm-4.46.0-pyh9f0ad1d_0.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/wcwidth-0.1.9-pyh9f0ad1d_0.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/webencodings-0.5.1-py_1.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/werkzeug-1.0.1-pyh9f0ad1d_0.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/zipp-3.1.0-py_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/absl-py-0.9.0-py36_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/asn1crypto-1.3.0-py36h9f0ad1d_1.tar.bz2
https://conda.anaconda.org/rdkit/linux-64/boost-1.63.0-py36h415b752_1.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/certifi-2020.4.5.1-py36h9f0ad1d_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/cffi-1.14.0-py36hd463f26_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/chardet-3.0.4-py36h9f0ad1d_1006.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/conda-package-handling-1.6.0-py36h8c4c3a4_2.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/cycler-0.10.0-py_2.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/cython-0.29.19-py36h831f99a_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/entrypoints-0.3-py36h9f0ad1d_1001.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/html5lib-0.9999999-py36_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/importlib-metadata-1.6.0-py36h9f0ad1d_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/jedi-0.17.0-py36h9f0ad1d_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/kiwisolver-1.2.0-py36hdb11119_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/llvmlite-0.32.0-py36hfa65bc7_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/markupsafe-1.1.1-py36h8c4c3a4_1.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/mistune-0.8.4-py36h8c4c3a4_1001.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/mock-4.0.2-py36h9f0ad1d_0.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/packaging-20.4-pyh9f0ad1d_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/pexpect-4.8.0-py36h9f0ad1d_1.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/pickleshare-0.7.5-py36h9f0ad1d_1001.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/pillow-5.0.0-py36_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/pycosat-0.6.3-py36h8c4c3a4_1004.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/pyrsistent-0.16.0-py36h8c4c3a4_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/pysocks-1.7.1-py36h9f0ad1d_1.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/python-dateutil-2.8.1-py_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/pyyaml-5.3.1-py36h8c4c3a4_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/pyzmq-19.0.1-py36h9947dbf_0.tar.bz2
https://repo.anaconda.com/pkgs/main/linux-64/qt-5.6.2-hd25b39d_14.conda
https://conda.anaconda.org/conda-forge/linux-64/ruamel_yaml-0.15.80-py36h8c4c3a4_1001.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/scipy-1.1.0-py36_blas_openblash7943236_201.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/simplejson-3.17.0-py36h8c4c3a4_1.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/tornado-6.0.4-py36h8c4c3a4_1.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/traitlets-4.3.3-py36h9f0ad1d_1.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/bleach-1.5.0-py36_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/cryptography-2.5-py36hb7f436b_1.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/importlib_metadata-1.6.0-0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/jupyter_core-4.6.3-py36h9f0ad1d_1.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/markdown-3.2.2-py_0.tar.bz2
https://conda.anaconda.org/omnia/linux-64/openmm-7.4.2-py36_cuda101_rc_1.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/pandas-0.22.0-py36_1.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/patsy-0.5.1-py_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/pyqt-5.6.0-py36h13b7fb3_1008.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/scikit-learn-0.19.1-py36_blas_openblas_201.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/setuptools-47.1.1-py36h9f0ad1d_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/terminado-0.8.3-py36h9f0ad1d_1.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/xgboost-0.6a2-py36_2.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/anyconfig-0.9.10-py_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/grpcio-1.16.0-py36h4f00d22_1000.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/jinja2-2.11.2-pyh9f0ad1d_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/jsonschema-3.2.0-py36h9f0ad1d_1.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/jupyter_client-6.1.3-py_0.tar.bz2
https://repo.anaconda.com/pkgs/main/linux-64/matplotlib-2.2.2-py36h0e671d2_0.conda
https://conda.anaconda.org/conda-forge/noarch/networkx-2.1-py_1.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/numba-0.49.1-py36h830a2c2_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/numexpr-2.7.1-py36h830a2c2_1.tar.bz2
https://conda.anaconda.org/omnia/linux-64/pdbfixer-1.4-py36_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/pluggy-0.13.1-py36h9f0ad1d_1.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/protobuf-3.12.1-py36h831f99a_0.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/pygments-2.6.1-py_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/pyopenssl-19.0.0-py36_0.tar.bz2
https://conda.anaconda.org/rdkit/linux-64/rdkit-2017.09.1-py36_1.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/statsmodels-0.10.2-py36hc1659b7_0.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/wheel-0.34.2-py_1.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/molvs-0.1.1-py_0.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/mordred-1.2.0-py_1.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/nbformat-5.0.6-py_0.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/pip-20.1.1-py_1.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/prompt-toolkit-3.0.5-py_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/pytables-3.6.1-py36h7b0bd57_2.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/pytest-5.4.2-py36h9f0ad1d_0.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/seaborn-0.10.0-py_0.tar.bz2
https://conda.anaconda.org/deepchem/noarch/simdna-0.4.2-py_0.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/swagger-spec-validator-2.5.0-py_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/tensorboard-1.6.0-py36_0.tar.bz2
https://repo.anaconda.com/pkgs/main/linux-64/tensorflow-gpu-base-1.6.0-py36hcdda91b_1.conda
https://conda.anaconda.org/conda-forge/linux-64/umap-learn-0.4.2-py36h9f0ad1d_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/urllib3-1.22-py36_0.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/bravado-core-5.16.0-py_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/ipython-7.15.0-py36h9f0ad1d_0.tar.bz2
https://conda.anaconda.org/deepchem/linux-64/mdtraj-1.9.1-py36_1.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/nbconvert-5.6.1-py36h9f0ad1d_1.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/prompt_toolkit-3.0.5-0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/requests-2.18.4-py36_1.tar.bz2
https://repo.anaconda.com/pkgs/main/linux-64/tensorflow-gpu-1.6.0-0.conda
https://conda.anaconda.org/conda-forge/noarch/bravado-10.4.3-py_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/conda-4.8.3-py36h9f0ad1d_1.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/ipykernel-5.3.0-py36h95af2a2_0.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/jupyter_console-6.1.0-py_1.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/notebook-6.0.3-py36h9f0ad1d_0.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/qtconsole-4.7.4-pyh9f0ad1d_0.tar.bz2
https://conda.anaconda.org/conda-forge/linux-64/widgetsnbextension-3.5.1-py36_0.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/ipywidgets-7.5.1-py_0.tar.bz2
https://conda.anaconda.org/conda-forge/noarch/jupyter-1.0.0-py_2.tar.bz2
https://conda.anaconda.org/deepchem/linux-64/deepchem-gpu-2.1.0-py36_0.tar.bz2
EOF

/content/AMPL/bin/conda install --file AMPL.txt -y

mkdir github
cd github
git clone https://github.com/ATOMconsortium/AMPL.git

cat << "EOF" > transformations_py.patch
--- transformations.py  2020-09-14 17:08:22.225747322 -0700
+++ transformations_patched.py  2020-09-14 17:08:07.869651225 -0700
@@ -9,7 +9,7 @@

 import numpy as np
 import pandas as pd
-import umap
+# import umap

 import deepchem as dc
 from deepchem.trans.transformers import Transformer, NormalizationTransformer
EOF

patch -N /content/github/AMPL/atomsci/ddm/pipeline/transformations.py transformations_py.patch

cat << "EOF" > __init___py.patch
--- /content/AMPL/atomsci/ddm/__init__.py.backup    2020-09-19 18:10:05.264013977 +0000
+++ /content/AMPL/atomsci/ddm/__init__.py   2020-09-19 18:15:37.338771924 +0000
@@ -1,6 +1,6 @@
 import pkg_resources
 try:
     __version__ = pkg_resources.require("atomsci-ampl")[0].version
-except TypeError:
+except:
     pass
EOF

patch -N /content/github/AMPL/atomsci/ddm/__init__.py __init___py.patch

PATH=/content/AMPL/bin:$PATH
PYTHONPATH=

cd /content/github/AMPL
./build.sh
./install.sh system
