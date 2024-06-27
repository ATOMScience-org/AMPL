.. _tests:

Tests
=====

`AMPL <https://github.com/ATOMScience-org/AMPL>`_ includes a suite of software tests. This section explains how to run a very simple test that is fast to run. The Python` test fits a random forest model using Mordred descriptors on a set of compounds from `Delaney, et al` with solubility data. A molecular scaffold-based split is used to create the training and test sets. In addition, an external holdout set is used to demonstrate how to make predictions on new compounds.

To run the `Delaney` Python script that curates a dataset, fits a model, and makes predictions, run the following commands:
::

    source atomsci/bin/activate # activate your atomsci pip environment
    cd atomsci/ddm/test/integrative/delaney_RF
    pytest

.. note:: 
   This test generally takes a few minutes on a modern system
 
 
The important files for this test are listed below:

* `test_delaney_RF.py`: This script loads and curates the dataset, generates a model pipeline object, and fits a model. The model is reloaded from the filesystem and then used to predict solubilities for a new dataset.
* `config_delaney_fit_RF.json`: Basic parameter file for fitting
* `config_delaney_predict_RF.json`: Basic parameter file for predicting  

More example and test information
---------------------------------
More details on examples and tests can be found in :ref:`Advanced testing <advanced_testing>`.  
