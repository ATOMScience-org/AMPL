######################
01 Install From Docker
######################

*Published: Nov, 2023, ATOM DDM Team*

------------

AMPL can be run from Docker to provide its accessibility across multiple platforms. This can be done from one of these options:


* :ref:`Pull an existing AMPL image from the Docker repo <step-2>`
* :ref:`Build a local image using Dockerfile. <create-local>`

1. Download and install Docker Desktop
======================================

Follow directions found here: https://www.docker.com/get-started.

.. _step-2:

2. Pull an existing AMPL image from Docker repo
===============================================

.. code-block:: bash

   docker pull atomsci/atomsci-ampl:latest

.. _step-3:

3. Run the AMPL image interactively
===================================

.. code-block:: bash

   docker run -it -p 8888:8888 -v </local_workspace_folder>:</directory_in_docker> atomsci/atomsci-ampl

For example

.. code-block:: bash

   docker run -it -p 8888:8888 -v ~:/home atomsci/atomsci-ampl

.. _step-4:

4. When inside the container, start the jupyter notebook
========================================================

.. code-block:: bash

      jupyter-notebook --ip=0.0.0.0 --allow-root --port=8888 &

This will output these messages:

.. code-block:: bash

       To access the server, open this file in a browser:
           file:///root/.local/share/jupyter/runtime/jpserver-14-open.html
       Or copy and paste one of these URLs:
           http://43aadd0c29db:8888/tree?token=b38528f4614743bdcac6e02c07cffabddd285007769d7d58
           http://127.0.0.1:8888/tree?token=b38528f4614743bdcac6e02c07cffabddd285007769d7d58

5. Go to a browser and type in the URL
======================================

Copy and paste the url containing ``127.0.0.1`` into your browser. For example:

.. code-block:: bash

   http://127.0.0.1:8888/tree?token=b38528f4614743bdcac6e02c07cffabddd285007769d7d58

.. note::
   
   If this doesn't work, exit the container and change port from 8888 to some other number such as ``7777`` or ``8899`` (in all 3 places it's written), then rerun both commands in :ref:`step 3 <step-3>` and :ref:`step 4 <step-4>`.  Be sure to save any work you want to be permanent in your workspace folder. If the container is shut down, you'll lose anything not in that folder.


The AMPL code is in:

.. code-block::

   http://127.0.0.1:<port_number>/tree/AMPL/atomsci/ddm/

6. Code examples
=================

The tutorials examples are in:

.. code-block::

   http://localhost:8888/tree/AMPL/atomsci/ddm/examples/tutorials2023

There are also examples in `AMPL's Read the Docs <https://ampl.readthedocs.io/en/latest/>`_ on how to use AMPL Framework.

----

7. To select an environment for the notebook, select ``venv`` as the run environment
====================================================================================

There are two ways to set an environment:


* From a notebook, top menu bar ``Kernel`` > ``Change Kernel`` > ``venv``


.. image:: ../_static/img/01_install_from_docker_files/docker_notebook_env2.png
   :target: ../_static/img/01_install_from_docker_files/docker_notebook_env2.png
   :alt: Select an environment from a notebook



* Outside of a notebook, click ``New`` dropdown from upper right corner, and select ``venv`` as the run environment


.. image:: ../_static/img/01_install_from_docker_files/docker_notebook_env1.png
   :target: ../_static/img/01_install_from_docker_files/docker_notebook_env1.png
   :alt: Select an environment outside of a notebook

.. _create-local:

Create a local image using ``Dockerfile``
========================================

AMPL `Dockerfile <../../../docker/Dockerfile>`_ is in ``AMPL/docker`` directory. To build a Docker image:

.. code-block:: bash

   docker build -t atomsci-ampl:<tag> .

Once it's built, follow the steps starting :ref:`step 3 <step-3>` to start and run the local copy of AMPL docker image.

Useful Docker commands
======================

.. code-block:: bash

   docker ps -a                                # check docker processes
   docker cp file.txt <container_id>:/file.txt # copy from local to container
   docker cp <container_id>:/file.txt file.txt # copy from container to local
