.. _install:

Installation
============

Python Environment
------------------

AMPL requires Python 3.10.

For LLNL users on LC::

    module load python/3.10.8

Create and activate a virtual environment::

    make sync-<platform>                  # for example: make sync-cpu
    source .venv-<platform>/bin/activate

.. note::

   Depending on system performance, creating the environment can take some time.

Install AMPL from PyPI
----------------------

AMPL is published to PyPI, so most users can install it directly without cloning the repository or building it locally::

    uv pip install atomsci-ampl

This installs the released AMPL package into your active virtual environment.

Install AMPL from a Local Clone
-------------------------------

If you want to work from the source code, first clone the repository::

    git clone https://github.com/ATOMScience-org/AMPL.git
    cd AMPL

Then choose one of the following workflows.

Development Install
~~~~~~~~~~~~~~~~~~~

For local development, install AMPL from the source tree in editable mode::

    ./install-dev.sh

This is the recommended workflow for development. Changes to the local source code are picked up without reinstalling the package.

Build and Install Locally
~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to build the package locally and install the built artifact::

    ./build.sh
    ./install.sh

In this workflow:

* ``build.sh`` builds the package artifacts locally.
* ``install.sh`` installs the locally built package into the active environment.