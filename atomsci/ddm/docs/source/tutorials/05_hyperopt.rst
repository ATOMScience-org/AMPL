##############################
05 Hyperparameter Optimization
##############################

*Published: June, 2024, ATOM DDM Team*

------------

Hyperparameters dictate the parameters of the training process and the
architecture of the model itself. For example, the number of random
trees is a hyperparameter for a **random forest**. In contrast, a
learned parameter for a **random forest** is the set of features that is
contained in a single node (in a single tree) and the cutoff values for
each of those features that determines how the data is split at that
node. A full discussion of hyperparameter optimization can be found on
`Wikipedia <https://en.wikipedia.org/wiki/Hyperparameter_optimization>`_.

The choice of hyperparameters strongly influences model performance, so
it is important to be able to optimize them as well.
`AMPL <https://github.com/ATOMScience-org/AMPL>`_ offers a variety
of hyperparameter optimization methods including random sampling, grid
search, and Bayesian optimization. Please refer to the parameter
documentation
`page <https://github.com/ATOMScience-org/AMPL#hyperparameter-optimization>`_
for further information.

In this tutorial we demonstrate the following: 

-  Build a parameter dictionary to perform a hyperparameter search for a **random forest** using Bayesian optimization. 
-  Perform the optimization process. 
-  Review the results

We will use these `AMPL <https://github.com/ATOMScience-org/AMPL>`_
functions: 

-  `parse_params <https://ampl.readthedocs.io/en/latest/utils.html#utils.hyperparam_search_wrapper.parse_params>`_
-  `build_search <https://ampl.readthedocs.io/en/latest/utils.html#utils.hyperparam_search_wrapper.build_search>`_
-  `run_search <https://ampl.readthedocs.io/en/latest/utils.html#utils.hyperparam_search_wrapper.HyperOptSearch.run_search>`_
-  `get_filesystem_perf_results <https://ampl.readthedocs.io/en/latest/pipeline.html#pipeline.compare_models.get_filesystem_perf_results>`_

The first three functions in the above list come from the
``hyperparameter_search_wrapper`` module.

Set Up Directories
******************

Here we set up a few important variables corresponding to required
directories and specific features for the **hyperparameter optimization
(HPO)** process. Then, we ensure that the directories are created before
saving models into them.

.. list-table::
   :header-rows: 1
   :class: tight-table

   * - Variable
     - Description
   * - `dataset_key`
     - The relative path to the dataset you want to use for HPO
   * - `descriptor_type`  
     - The type of features you want to use during HPO
   * - `model_dir`
     - The directory where you want to save all of the models
   * - `best_model_dir`
     - For Bayesian optimization, the winning model is saved in this separate folder
   * - `split_uuid`
     - The presaved split uuid from **Tutorial 3, "Splitting Datasets for Validation and Testing"**

.. code:: ipython3

    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    import os
    
    dataset_key='dataset/SLC6A3_Ki_curated.csv'
    descriptor_type = 'rdkit_raw'
    model_dir = 'dataset/SLC6A3_models'
    best_model_dir = 'dataset/SLC6A3_models/best_models'
    split_uuid = "c35aeaab-910c-4dcf-8f9f-04b55179aa1a"
    
    
    if not os.path.exists(f'./{best_model_dir}'):
        os.mkdir(f'./{best_model_dir}')
        
    if not os.path.exists(f'./{model_dir}'):
        os.mkdir(f'./{model_dir}')

To run a hyperparameter search, we first create a parameter dictionary
with parameter settings that will be common to all models, along with
some special parameters that control the search and indicate which
parameters will be varied and how. The table below describes the special
parameter settings for our random forest search.

Parameter Dictionary Settings
*****************************

.. list-table::
   :header-rows: 1
   :class: tight-table

   * - Parameter
     - Description
   * - `'hyperparam':'True'`
     - This setting indicates that we are performing a hyperparameter search instead of just training one model.
   * - `'previously_featurized':True'`
     - This tells `AMPL <https://github.com/ATOMScience-org/AMPL>`_ to search for previously generated features in ../dataset/scaled_descriptors instead of regenerating them on the fly.
   * - `'search_type':'hyperopt'`
     - This specifies the hyperparameter search method. Other options include grid, random, and geometric. Specifications for each hyperparameter search method is different, please refer to the full documentation. Here we are using the Bayesian optimization method.
   * - `'model_type':'RF|10'`
     - This means `AMPL <https://github.com/ATOMScience-org/AMPL>`_ will try 10 times to find the best set of hyperparameters using **random forests**. In practice, this parameter could be set to 100 or more.
   * - `'rfe':'uniformint|8,512'`
     - The Bayesian optimizer will uniformly search between 8 and 512 for the best number of random forest estimators. Similarly rfd stands for **random forest depth** and ``rff`` stands for **random forest features**.
   * - `'result_dir'`
     - Now expects two parameters. The first directory will contain the best trained models while the second directory will contain all models trained in the search.

Regression models are optimized to maximize the :math:`R^2` and
classification models are optimized using area under the receiver
operating characteristic curve. A full list of parameters can be found
on our
`github <https://github.com/ATOMScience-org/AMPL/blob/master/atomsci/ddm/docs/PARAMETERS.md>`_.

.. code:: ipython3

    params = {
        "hyperparam": "True",
        "prediction_type": "regression",
    
        "dataset_key": dataset_key,
        "id_col": "compound_id",
        "smiles_col": "base_rdkit_smiles",
        "response_cols": "avg_pKi",
    
        "splitter":"scaffold",
        "split_uuid": split_uuid,
        "previously_split": "True",
    
        "featurizer": "computed_descriptors",
        "descriptor_type" : descriptor_type,
        "transformers": "True",
    
        "search_type": "hyperopt",
        "model_type": "RF|10",
        "rfe": "uniformint|8,512",
        "rfd": "uniformint|6,32",
        "rff": "uniformint|8,200",
    
        "result_dir": f"./{best_model_dir},./{model_dir}"
    }

Run Hyperparameter Search
*************************

In **Tutorial 3, "Train a Simple Regression Model"**, we directly
imported the ``parameter_parser`` and ``model_pipeline`` objects to
parse the ``config`` dict and train a single model. Here, we use
``hyperparameter_search_wrapper`` to handle many models for us. First we
build the search by creating a list of parameters to use, and then we
run the search.

.. code:: ipython3

    import atomsci.ddm.utils.hyperparam_search_wrapper as hsw
    import importlib
    importlib.reload(hsw)
    ampl_param = hsw.parse_params(params)
    hs = hsw.build_search(ampl_param)
    hs.run_search()


The top scoring model will be saved in
``dataset/SLC6A3_models/best_models`` along with a csv file containing
regression performance for all trained models.

All of the models are saved in ``dataset/SLC6A3_models``. These models
can be explored using ``get_filesystem_perf_results``. A full analysis
of the hyperparameter performance is explored in **Tutorial 6, "Compare
models to select the best hyperparameters"**.

.. code:: ipython3

    import atomsci.ddm.pipeline.compare_models as cm
    
    result_df = cm.get_filesystem_perf_results(
        result_dir=model_dir,
        pred_type='regression'
    )
    
    # sort by validation r2 score to see top performing models
    result_df = result_df.sort_values(by='best_valid_r2_score', ascending=False)
    result_df[['model_uuid','model_parameters_dict','best_valid_r2_score','best_test_r2_score']].head()


.. parsed-literal::

    Found data for 10 models under dataset/SLC6A3_models


.. list-table::
   :header-rows: 1
   :class: tight-table
  
   * -                                     
     - model_uuid                      
     - model_parameters_dict
     - best_valid_r2_score
     - best_test_r2_score
   * - **4**
     - dbd1d89c-05f5-4224-bce4-7dbeafaba313
     - {"rf_estimators": 262, "rf_max_depth": 16, "rf...
     - 0.488461
     - 0.424234
   * - 8
     - 601ae89f-a8bb-4da2-b7a7-b434a2bdcbbe
     - {"rf_estimators": 190, "rf_max_depth": 15, "rf...
     - 0.483822
     - 0.448591
   * - 9
     - 0967e5ea-64a1-4509-80da-176bd8773775
     - {"rf_estimators": 146, "rf_max_depth": 27, "rf...
     - 0.483401
     - 0.436227
   * - 2
     - 9da5fa7a-610f-469a-9562-b760c03581bc
     - {"rf_estimators": 60, "rf_max_depth": 28, "rf_...
     - 0.480939
     - 0.450400
   * - 1
     - 2b63bedb-7983-49cd-8d9b-b2039439ae98
     - {"rf_estimators": 233, "rf_max_depth": 28, "rf...
     - 0.480583
     - 0.399987



Examples of Other Parameter Sets
*****************************

Below are some parameters that can be used for **neural networks**,
`XGBoost <https://en.wikipedia.org/wiki/XGBoost>`_ models,
**fingerprint splits** and
`ECFP <https://pubs.acs.org/doi/10.1021/ci100050t>`_ features. Each
set of parameters can be used to replace the parameters above. Trying
them out is left as an exercise for the reader.

Neural Network Hyperopt Search
------------------------------

.. list-table::
   :header-rows: 1
   :class: tight-table
  
   * - Parameter                                     
     - Description   
   * - `lr`
     - This controls the learning rate. loguniform|-13.8,-3 means the logarithm of the learning rate is uniformly distributed between -13.8 and -3.
   * - `ls`
     - This controls layer sizes. 3|8,512 means 3 layers with sizes ranging between 8 and 512 neurons. A good strategy is to start with a fewer layers and slowly increase the number until performance plateaus.
   * - `dp`
     - This controls dropout. 3|0,0.4 means 3 dropout layers with probability of zeroing a weight between 0 and 40%. This needs to match the number of layers specified with `ls` and should range between 0% and 50%.
   * - `max_epochs`
     - This controls how long to train each model. Training for more epochs increases runtime, but allows models more time to optimize.

.. code:: ipython3

    params = {
        "hyperparam": "True",
        "prediction_type": "regression",
    
        "dataset_key": dataset_key,
        "id_col": "compound_id",
        "smiles_col": "base_rdkit_smiles",
        "response_cols": "avg_pKi",
    
        "splitter":"scaffold",
        "split_uuid": split_uuid,
        "previously_split": "True",
    
        "featurizer": "computed_descriptors",
        "descriptor_type" : descriptor_type,
        "transformers": "True",
    
        ### Use a NN model
        "search_type": "hyperopt",
        "model_type": "NN|10",
        "lr": "loguniform|-13.8,-3",
        "ls": "uniformint|3|8,512",
        "dp": "uniform|3|0,0.4",
        "max_epochs":100,
        ###
    
        "result_dir": f"./{best_model_dir},./{model_dir}"
    }

XGBoost
-------

-  ``xgbg`` Stands for ``xgb_gamma`` and controls the minimum loss
   reduction required to make a further partition on a leaf node of the
   tree.
-  ``xgbl`` Stands for ``xgb_learning_rate`` and controls the boosting
   learning rate searching domain of
   `XGBoost <https://en.wikipedia.org/wiki/XGBoost>`_ models.

.. code:: ipython3

    params = {
        "hyperparam": "True",
        "prediction_type": "regression",
    
        "dataset_key": dataset_key,
        "id_col": "compound_id",
        "smiles_col": "base_rdkit_smiles",
        "response_cols": "avg_pKi",
    
        "splitter":"scaffold",
        "split_uuid": split_uuid,
        "previously_split": "True",
    
        "featurizer": "computed_descriptors",
        "descriptor_type" : descriptor_type,
        "transformers": "True",
    
        ### Use an XGBoost model
        "search_type": "hyperopt",
        "model_type": "xgboost|10",
        "xgbg": "uniform|0,0.2",
        "xgbl": "loguniform|-2,2",
        ###
    
        "result_dir": f"./{best_model_dir},./{model_dir}"
    }

Fingerprint Split
-----------------

This trains an `XGBoost <https://en.wikipedia.org/wiki/XGBoost>`_
model using a provided **fingerprint split**.

.. code:: ipython3

    fp_split_uuid="be60c264-6ac0-4841-a6b6-41bf846e4ae4"
    
    params = {
        "hyperparam": "True",
        "prediction_type": "regression",
    
        "dataset_key": dataset_key,
        "id_col": "compound_id",
        "smiles_col": "base_rdkit_smiles",
        "response_cols": "avg_pKi",
    
        ### Use a fingerprint split
        "splitter":"fingerprint",
        "split_uuid": fp_split_uuid,
        "previously_split": "True",
        ###
    
        "featurizer": "computed_descriptors",
        "descriptor_type" : descriptor_type,
        "transformers": "True",
    
        "search_type": "hyperopt",
        "model_type": "xgboost|10",
        "xgbg": "uniform|0,0.2",
        "xgbl": "loguniform|-2,2",
    
        "result_dir": f"./{best_model_dir},./{model_dir}"
    }

ECFP Features
-------------

This uses an `XGBoost <https://en.wikipedia.org/wiki/XGBoost>`_
model with `ECFP
fingerprints <https://pubs.acs.org/doi/10.1021/ci100050t>`_ features
and a **scaffold split**.

.. code:: ipython3

    params = {
        "hyperparam": "True",
        "prediction_type": "regression",
    
        "dataset_key": dataset_key,
        "id_col": "compound_id",
        "smiles_col": "base_rdkit_smiles",
        "response_cols": "avg_pKi",
    
        "splitter":"scaffold",
        "split_uuid": split_uuid,
        "previously_split": "True",
    
        ### Use ECFP Features
        "featurizer": "ecfp",
        "ecfp_radius" : 2,
        "ecfp_size" : 1024,
        "transformers": "True",
        ###
    
        "search_type": "hyperopt",
        "model_type": "xgboost|10",
        "xgbg": "uniform|0,0.2",
        "xgbl": "loguniform|-2,2",
    
        "result_dir": f"./{best_model_dir},./{model_dir}"
    }

In **Tutorial 6, "Compare Models to Select the Best Hyperparameters"**,
we analyze the performance of these large sets of models to select the
best hyperparameters for production models.

If you have specific feedback about a tutorial, please complete the
`AMPL Tutorial Evaluation <https://forms.gle/pa9sHj4MHbS5zG7A6>`_.
