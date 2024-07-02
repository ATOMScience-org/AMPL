.. _advanced_installation:

Advanced Installation
=====================

Deployment
----------
`AMPL <https://github.com/ATOMScience-org/AMPL>`_ has been developed and tested on the following Linux systems:

* Red Hat Enterprise Linux 7 with SLURM
* Ubuntu 16.04
 
Uninstallation
--------------
To remove AMPL from a pip environment use:
::

    deactivate
    pip uninstall atomsci-ampl
 

To remove the atomsci pip environment entirely from a system use:
::

   deactivate
   cd $parent # to the parent of the atomsci `pip env` dir
   rm -r atomsci
