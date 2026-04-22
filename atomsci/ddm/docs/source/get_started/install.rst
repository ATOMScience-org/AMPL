.. _install:

Installation
============

Setup Repo, Pip Environment
---------------------------

Clone git repository::

    git clone https://github.com/ATOMScience-org/AMPL.git


Create a virtual environment::

    module load python/3.10.8             # use python 3.10.8
    make sync-<platform>                  # for example sync-cpu
    source .venv-<platform>/bin/activate  # to activate your environment

To install AMPL python package::

    python -m pip install --index-url https://pypi.org/simple atomsci-ampl

.. note::

   Depending on system performance, creating the environment can take some time.