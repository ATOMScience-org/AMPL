.. _install:

Installation
============

Setup Repo, Pip Environment
---------------------------

Clone git repository::

    git clone https://github.com/ATOMScience-org/AMPL.git
 
Please refer to this `install link <https://github.com/ATOMScience-org/AMPL#Install>`_, for details.

Create pip environment::

    module load python/3.9.12 # use python 3.9.12
    python3 -m venv atomsci-env # create a new pip env
    source atomsci-env/bin/activate # activate the environemt

    python3 -m pip install pip --upgrade
    cd $AMPL_HOME/pip # cd to AMPL repo's pip directory

    pip3 install --force-reinstall -r requirements.txt

.. note::

   Depending on system performance, creating the environment can take some time.

Build and Install AMPL
----------------------
Go to the AMPL root directory and install the `AMPL <https://github.com/ATOMScience-org/AMPL>`_  package::

    source atomsci-env/bin/activate # activate the environemt
    cd ..
    ./build.sh
    pip3 install -e .

* The `install.sh` system command installs AMPL directly in the pip environment. If `install.sh` alone is used, then AMPL is installed in the `$HOME/.local` directory.

* After this process, you will have an `atomsci-env` pip environment with all dependencies installed. The name of the AMPL package is `atomsci-ampl` and is installed in the `install.sh` script to the environment with pip.  
