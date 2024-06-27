.. ATOM Data-Driven Modeling Pipeline documentation master file, created by
   sphinx-quickstart on Wed Jul 31 10:53:25 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ATOM Modeling PipeLine (AMPL) for Drug Discovery
=================================================

.. include:: /_static/shared/links.rst

`AMPL <https://github.com/ATOMScience-org/AMPL>`_ is an open-source, modular, extensible software pipeline for building and sharing models to advance in silico drug discovery.

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

- `DeepChem <DeepChem_>`_: The basis for the graph convolution models
- `RDKit <RDKit_>`_: Molecular informatics library
- `Mordred <Mordred_>`_: Chemical descriptors
- Other Python package dependencies

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   get_started/introduction
   get_started/install
   get_started/install_with_docker
   get_started/running_ampl
   get_started/tests

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/ampl_tutorials_intro
   tutorials/01_data_curation
   tutorials/02_perform_a_split
   tutorials/03_train_regression_model
   tutorials/04_application_of_a_trained_model
   tutorials/05_hyperopt
   tutorials/06_compare_models
   tutorials/07_train_a_production_model
   tutorials/08_visualizations_of_model_performance

.. toctree::
   :maxdepth: 2
   :caption: Advanced Concepts

   advanced/advanced_ampl_usage
   advanced/advanced_installation
   advanced/advanced_testing

.. toctree::
   :maxdepth: 1
   :caption: API Package Reference

   modules

.. toctree::
   :maxdepth: 1
   :caption: External Resources

   external/resources
