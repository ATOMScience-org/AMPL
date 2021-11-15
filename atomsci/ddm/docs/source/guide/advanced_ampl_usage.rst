.. advanced_ampl_usage:

Advanced AMPL Usage
===================

Command line
------------
**AMPL** can `fit` models from the command line with:
::

    python model_pipeline.py --config_file test.json
 
Hyperparameter optimization
---------------------------
Hyperparameter optimization for AMPL model fitting is available to run on SLURM clusters or with `HyperOpt <https://hyperopt.github.io/hyperopt/>`_. (Bayesian Optimization). To run Bayesian Optimization, the following steps can be followed.

See `Hyperparameter optimization <https://github.com/ATOMScience-org/AMPL#hyperparameter-optimization>`_ for more details.
