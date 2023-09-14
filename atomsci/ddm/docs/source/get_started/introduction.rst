.. _getting_started:

Getting started
===============

Prerequisites
-------------
**AMPL** is a Python 3 package that has been developed and run in a specific pip environment.
 
Install
-------
Clone the git repository::

    git clone https://github.com/ATOMScience-org/AMPL.git
 
Please refer to this link, https://github.com/ATOMScience-org/AMPL#Install, for details.

Create pip environment::

    module load python/3.8.2 # use python 3.8.2
    python3 -m venv atomsci # create a new pip env
    source atomsci/bin/activate # activate the environemt

    python3 -m pip install pip --upgrade
    cd $AMPL_HOME/pip # cd to AMPL repo's pip directory

    pip3 install --force-reinstall --no-use-pep517 -r requirements.txt

.. note::

   Depending on system performance, creating the environment can take some time.
