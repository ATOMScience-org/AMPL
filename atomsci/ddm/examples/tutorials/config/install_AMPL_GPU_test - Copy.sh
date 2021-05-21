mkdir github
cd github
git clone https://github.com/ATOMconsortium/AMPL.git
cd AMPL
git checkout pkg_upgrade
cd ..

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
