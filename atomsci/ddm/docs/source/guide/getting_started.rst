.. _getting_started:

Getting started
===============

Prerequisites
-------------
**AMPL** is a Python 3 package that has been developed and run in a specific conda environment. The following prerequisites are necessary to install AMPL:

* conda (Anaconda 3 or Miniconda 3, Python 3)
 
Install
-------
Clone the git repository::

    git clone https://github.com/ATOMScience-org/AMPL.git
 

Create conda environment::

    cd conda
    conda create -y -n atomsci --file conda_package_list.txt
    conda activate atomsci
    pip install -r pip_requirements.txt

.. note::

   Depending on system performance, creating the environment can take some time.
