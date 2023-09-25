.. ATOM Data-Driven Modeling Pipeline documentation master file, created by
   sphinx-quickstart on Wed Jul 31 10:53:25 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ATOM Modeling PipeLine (AMPL) for Drug Discovery
==============================================================

**AMPL** is an open-source, modular, extensible software pipeline for building and sharing models to advance in silico drug discovery.

The ATOM Modeling PipeLine (AMPL) extends the functionality of DeepChem and supports an array of machine learning and molecular featurization tools. AMPL is an end-to-end data-driven modeling pipeline to generate machine learning models that can predict key safety and pharmacokinetic-relevant parameters. AMPL has been benchmarked on a large collection of pharmaceutical datasets covering a wide range of parameters.

Features
--------
AMPL enables tasks for modeling and prediction from data ingestion to data analysis and can be broken down into the following stages:

- Data ingestion and curation
- Featurization
- Model training and tuning
- Prediction generation
- Visualization and analysis
- Details of running specific features are within the parameter (options) documentation. 

More detailed documentation is in the library documentation.

Built with
----------

- `DeepChem`_: The basis for the graph convolution models
- `RDKit`_: Molecular informatics library
- `Mordred`_: Chemical descriptors
- Other Python package dependencies

Get Started
-----------

A step-by-step guide to getting started with MolVS.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   get_started/introduction
   get_started/install
   get_started/run_ampl
   get_started/tests

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/apply_trained_model
   tutorials/cla_scaff_cleanup
   tutorials/cla_scaff
   tutorials/compare_models
   tutorials/curate_datasets
   tutorials/docker
   tutorials/hyperparameter_search
   tutorials/perform_split
   tutorials/production_model
   tutorials/train_simple_model
   tutorials/visualization

.. toctree::
   :maxdepth: 1
   :caption: API Package Reference

   modules

.. toctree::
   :maxdepth: 2
   :caption: Advanced Concepts

   advanced/advanced_ampl_usage
   advanced/advanced_installation
   advanced/advanced_testing

Useful links
------------
- `ATOM Data-Driven Modeling Pipeline on GitHub`_
- `Pipeline parameters (options)`_
- `Library documentation`_

.. _`DeepChem`: https://github.com/deepchem/deepchem
.. _`RDKit`: http://www.rdkit.org
.. _`Mordred`: https://github.com/mordred-descriptor/mordred
.. _`ATOM Data-Driven Modeling Pipeline on GitHub`: https://github.com/ATOMScience-org/AMPL
.. _`Pipeline parameters (options)`: https://github.com/ATOMScience-org/AMPL/blob/master/atomsci/ddm/docs/PARAMETERS.md
.. _`Library documentation`: https://ampl.readthedocs.io/en/latest/index.html
