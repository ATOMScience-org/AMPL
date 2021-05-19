.. _advanced_installation:

Advanced Installation
=====================

Deployment
----------
**AMPL** has been developed and tested on the following Linux systems:

* Red Hat Enterprise Linux 7 with SLURM
* Ubuntu 16.04
 
Uninstallation
--------------
To remove AMPL from a conda environment use:
::

    conda activate atomsci
    pip uninstall atomsci-ampl
 

To remove the atomsci conda environment entirely from a system use:
::

   conda deactivate
   conda remove --name atomsci --all
