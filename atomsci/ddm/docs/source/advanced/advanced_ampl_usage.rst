.. advanced_ampl_usage:

Advanced AMPL Usage
===================

.. include:: /_static/shared/links.rst

Command line
------------
`AMPL <https://github.com/ATOMScience-org/AMPL>`_ can `fit` models from the command line with:
::

    python model_pipeline.py --config_file test.json
 
Hyperparameter optimization
---------------------------
Hyperparameter optimization for AMPL model fitting is available to run on SLURM clusters or with `HyperOpt <HyperOpt_>`_. (Bayesian Optimization). To run Bayesian Optimization, the following steps can be followed.

See `Hyperparameter optimization <Hyperparameter optimization_>`_ for more details.
