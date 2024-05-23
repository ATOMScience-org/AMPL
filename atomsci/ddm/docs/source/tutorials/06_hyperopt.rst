##############################
06 Hyperparameter Optimization
##############################

*Published: May, 2024, ATOM DDM Team*

------------

In this tutorial we demonstrate the following: - Build a parameter
dictionary to perform a ``hyperparameter optimization`` for a random
forest using ``Bayesian optimization``. 

-  Perform the optimization process. 
-  Review the results

We will use these |ampl| functions here:

-  `parse\_params <https://ampl.readthedocs.io/en/latest/utils.html#utils.hyperparam_search_wrapper.parse_params>`_
-  `build\_search <https://ampl.readthedocs.io/en/latest/utils.html#utils.hyperparam_search_wrapper.build_search>`_
-  `run\_search <https://ampl.readthedocs.io/en/latest/utils.html#utils.hyperparam_search_wrapper.HyperOptSearch.run_search>`_
-  `get\_filesystem\_perf\_results <https://ampl.readthedocs.io/en/latest/pipeline.html#pipeline.compare_models.get_filesystem_perf_results>`_

``Hyperparameters`` dictate the parameters of the training process and
the architecture of the model itself. For example, the number of random
trees is a hyperparameter for a random forest. In contrast, a learned
parameter for a random forest is the set of features that is contained
in a single node (in a single tree) and the cutoff values for each of
those features that determines how the data is split at that node. A
full discussion of hyperparameter optimization can be found on
|wiki|.

The choice for hyperparameters strongly influence model performance, so
it is important to be able to optimize them as well.
|ampl|  offers a variety
of hyperparameter optimization methods including random sampling, grid
search, and Bayesian optimization. Further information for
|ampl|'s
``Bayesian optimization`` can be found
|hyper_opt|.

Setup directories
*****************

Describe important features like descriptor type and output directories.
Make sure the directories are created before training the models.

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

Parameter dictionary settings.
******************************

-  ``'hyperparam':True`` This setting indicates that we are performing a
   hyperparameter search instead of just training one model.
-  ``'previously_featurized':'True'`` This tells AMPL to search for
   previously generated features in ``../dataset/scaled_descriptors``
   instead of regenerating them on the fly.
-  ``'search_type':'hyperopt'`` This specifies the hyperparameter search
   method. Other options include grid, random, and geometric.
   Specifications for each hyperparameter search method is different,
   please refer to the full documentation. Here we are using the
   ``Bayesian optimization`` method.
-  ``'model_type':'RF|10'`` This means
   |ampl|  will try 10
   times to find the best set of hyperparameters using random forests.
   In production this parameter could be set to 100 or more.
-  ``'rfe':'uniformint|8,512'`` The ``Bayesian optimizer`` will
   uniformly search between 8 and 512 for the best number of random
   forest estimators. Similarly ``rfd`` stands for random forest depth
   and ``rff`` stands for random forest features.
-  ``result_dir`` Now expects two parameters. The first directory will
   contain the best trained models while the second directory will
   contain all models trained in the search.

Regression models are optimized using root mean squared loss and
classification models are optimized using area under the receiver
operating characteristic curve. A full list of parameters can be found
on our github
|params|.

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

In **Tutorial 4, "Train a Simple Regression Model"** we directly
imported the ``parameter_parser`` and ``model_pipeline`` objects to
parse the config dict and train a single model. Here, we use
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


.. parsed-literal::

    model_performance|train_r2|train_rms|valid_r2|valid_rms|test_r2|test_rms|model_params|model
    
    rf_estimators: 65, rf_max_depth: 22, rf_max_feature: 33
    RF model with computed_descriptors and rdkit_raw      
      0%|          | 0/10 [00:00<?, ?trial/s, best loss=?]

.. parsed-literal::

    2024-04-16 11:19:29,471 Previous dataset split restored


.. parsed-literal::

    model_performance|0.948|0.284|0.463|0.885|0.385|0.955|65_22_33|./dataset/SLC6A3_models/SLC6A3_Ki_curated_model_65d93c86-11e8-4f79-a6be-384db6956d26.tar.gz
    
    rf_estimators: 233, rf_max_depth: 28, rf_max_feature: 12                        
    RF model with computed_descriptors and rdkit_raw                                
     10%|█         | 1/10 [00:00<00:06,  1.44trial/s, best loss: 0.5365818670592989]

.. parsed-literal::

    2024-04-16 11:19:30,177 Previous dataset split restored


.. parsed-literal::

    model_performance|0.948|0.284|0.481|0.871|0.400|0.944|233_28_12|./dataset/SLC6A3_models/SLC6A3_Ki_curated_model_2b63bedb-7983-49cd-8d9b-b2039439ae98.tar.gz
    
    rf_estimators: 60, rf_max_depth: 28, rf_max_feature: 73                         
    RF model with computed_descriptors and rdkit_raw                                
     20%|██        | 2/10 [00:02<00:09,  1.25s/trial, best loss: 0.5194165178690741]

.. parsed-literal::

    2024-04-16 11:19:31,809 Previous dataset split restored


.. parsed-literal::

    model_performance|0.947|0.287|0.481|0.871|0.450|0.903|60_28_73|./dataset/SLC6A3_models/SLC6A3_Ki_curated_model_9da5fa7a-610f-469a-9562-b760c03581bc.tar.gz
    
    rf_estimators: 158, rf_max_depth: 7, rf_max_feature: 92                         
    RF model with computed_descriptors and rdkit_raw                                
     30%|███       | 3/10 [00:03<00:06,  1.00trial/s, best loss: 0.5190614320716579]

.. parsed-literal::

    2024-04-16 11:19:32,512 Previous dataset split restored


.. parsed-literal::

    model_performance|0.836|0.503|0.471|0.879|0.418|0.929|158_7_92|./dataset/SLC6A3_models/SLC6A3_Ki_curated_model_4f36098e-a8fe-4469-922e-5dca432f355b.tar.gz
    
    rf_estimators: 262, rf_max_depth: 16, rf_max_feature: 40                        
    RF model with computed_descriptors and rdkit_raw                                
     40%|████      | 4/10 [00:04<00:06,  1.04s/trial, best loss: 0.5190614320716579]

.. parsed-literal::

    2024-04-16 11:19:33,614 Previous dataset split restored


.. parsed-literal::

    model_performance|0.948|0.285|0.488|0.864|0.424|0.924|262_16_40|./dataset/SLC6A3_models/SLC6A3_Ki_curated_model_dbd1d89c-05f5-4224-bce4-7dbeafaba313.tar.gz
    
    rf_estimators: 393, rf_max_depth: 28, rf_max_feature: 190                       
    RF model with computed_descriptors and rdkit_raw                                
     50%|█████     | 5/10 [00:05<00:06,  1.28s/trial, best loss: 0.5115391017103005]

.. parsed-literal::

    2024-04-16 11:19:35,308 Previous dataset split restored


.. parsed-literal::

    model_performance|0.950|0.277|0.476|0.875|0.428|0.921|393_28_190|./dataset/SLC6A3_models/SLC6A3_Ki_curated_model_8e7bb4a7-40ef-4400-8c9d-c07dbf496e56.tar.gz
    
    rf_estimators: 29, rf_max_depth: 23, rf_max_feature: 177                        
    RF model with computed_descriptors and rdkit_raw                                
     60%|██████    | 6/10 [00:08<00:07,  1.83s/trial, best loss: 0.5115391017103005]

.. parsed-literal::

    2024-04-16 11:19:38,210 Previous dataset split restored


.. parsed-literal::

    model_performance|0.946|0.288|0.471|0.879|0.427|0.922|29_23_177|./dataset/SLC6A3_models/SLC6A3_Ki_curated_model_4596c9af-f98c-4ce4-bb79-91fedb4c0ea6.tar.gz
    
    rf_estimators: 106, rf_max_depth: 10, rf_max_feature: 112                       
    RF model with computed_descriptors and rdkit_raw                                
     70%|███████   | 7/10 [00:09<00:04,  1.40s/trial, best loss: 0.5115391017103005]

.. parsed-literal::

    2024-04-16 11:19:38,736 Previous dataset split restored


.. parsed-literal::

    model_performance|0.914|0.366|0.474|0.876|0.414|0.932|106_10_112|./dataset/SLC6A3_models/SLC6A3_Ki_curated_model_67b2be27-3a1f-4e16-9d0a-2337e431907c.tar.gz
    
    rf_estimators: 190, rf_max_depth: 15, rf_max_feature: 135                       
    RF model with computed_descriptors and rdkit_raw                                
     80%|████████  | 8/10 [00:10<00:02,  1.21s/trial, best loss: 0.5115391017103005]

.. parsed-literal::

    2024-04-16 11:19:39,511 Previous dataset split restored


.. parsed-literal::

    model_performance|0.947|0.286|0.484|0.868|0.449|0.905|190_15_135|./dataset/SLC6A3_models/SLC6A3_Ki_curated_model_601ae89f-a8bb-4da2-b7a7-b434a2bdcbbe.tar.gz
    
    rf_estimators: 146, rf_max_depth: 27, rf_max_feature: 112                       
    RF model with computed_descriptors and rdkit_raw                                
     90%|█████████ | 9/10 [00:11<00:01,  1.28s/trial, best loss: 0.5115391017103005]

.. parsed-literal::

    2024-04-16 11:19:40,938 Previous dataset split restored


.. parsed-literal::

    model_performance|0.949|0.280|0.483|0.869|0.436|0.915|146_27_112|./dataset/SLC6A3_models/SLC6A3_Ki_curated_model_0967e5ea-64a1-4509-80da-176bd8773775.tar.gz
    
    100%|██████████| 10/10 [00:12<00:00,  1.27s/trial, best loss: 0.5115391017103005]
    Generating the performance -- iteration table and Copy the best model tarball.
    Best model: ./dataset/SLC6A3_models/SLC6A3_Ki_curated_model_dbd1d89c-05f5-4224-bce4-7dbeafaba313.tar.gz, valid R2: 0.4884608982896995


The top scoring model will be saved in
``dataset/SLC6A3_models/best_models`` along with a csv file containing
regression performance for all trained models.

All of the models are saved in ``dataset/SLC6A3_models``. These models
can be explored using ``get_filesystem_perf_results``. A full analysis
of the hyperparameter performance is explored in **Tutorial 7, "Compare
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
   :widths: 3 10 10 5 5
   :header-rows: 1
   :class: tight-table
  
   * -                                     
     - model_uuid                      
     - model_parameters_dict
     - best_valid_r2_score
     - best_test_r2_score
   * - 4
     - dbd1d89c-05f5-4224-bce4-7dbeafaba313
     - {"rf_estimators": 262, "rf_max_depth": 16, "rf...",...}
     - 0.488461
     - 0.424234
   * - 8
     - 601ae89f-a8bb-4da2-b7a7-b434a2bdcbbe
     - {"rf_estimators": 190, "rf_max_depth": 15, "rf...",...}
     - 0.483822
     - 0.448591
   * - 9
     - 0967e5ea-64a1-4509-80da-176bd8773775
     - {"rf_estimators": 146, "rf_max_depth": 27, "rf...",...}
     - 0.483401
     - 0.436227
   * - 2
     - 9da5fa7a-610f-469a-9562-b760c03581bc
     - {"rf_estimators": 60, "rf_max_depth": 28, "rf_...",...}
     - 0.480939
     - 0.450400
   * - 1
     - 2b63bedb-7983-49cd-8d9b-b2039439ae98
     - {"rf_estimators": 233, "rf_max_depth": 28, "rf...",...}
     - 0.480583
     - 0.399987


Examples for other parameters
=============================

Below are some parameters that can be used for neural networks,
|xgboost| models,
fingerprint splits and
|ecfp| features. Each
set of parameters can be used to replace the parameters above. Trying
them out is left as an exercise for the reader.

Neural Network Hyperopt Search
------------------------------

-  ``lr`` This controls the learning rate. ``loguniform|-13.8,-3`` means
   the logarithm of the learning rate is uniformly distributed between
   ``-13.8`` and ``-3``.
-  ``ls`` This controls layer sizes. ``3|8,512`` means 3 layers with
   sizes ranging between 8 and 512 neurons. A good strategy is to start
   with a fewer layers and slowly increase the number until performance
   plateaus.
-  ``dp`` This controls dropout. ``3|0,0.4`` means 3 dropout layers with
   probability of zeroing a weight between 0 and 40%. This needs to
   match the number of layers specified with ``ls`` and should range
   between 0% and 50%.
-  ``max_epochs`` This controls how long to train each model. Training
   for more epochs increases runtime, but allows models more time to
   optimize.

::

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
        "max_epochs":100
        ###

        "result_dir": f"./{best_model_dir},./{model_dir}"
    }

XGBoost
-------

-  ``xgbg`` Stands for xgb\_gamma and controls the minimum loss
   reduction required to make a further partition on a leaf node of the
   tree.
-  ``xgbl`` Stands for xgb\_learning\_rate and controls the boosting
   learning rate searching domain of XGBoost models.

::

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

This trains an XGBoost model using a fingerprint split created in
**Tutorial 3, "Splitting Datasets for Validation and Testing"**.

::

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

This uses an XGBoost model with ECFP features and a scaffold split.

::

    fp_split_uuid="be60c264-6ac0-4841-a6b6-41bf846e4ae4"

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

In **tutorial 7**, we analyze the performance of these large sets of
models to select the best ``hyperparameters`` for ``production models``.

.. |ampl| raw:: html

   <b><a href="https://github.com/ATOMScience-org/AMPL">AMPL</a></b>


.. |wiki| raw:: html

   <b><a href="https://en.wikipedia.org/wiki/Hyperparameter_optimization">wikipedia</a></b>

.. |hyper_opt| raw:: html

   <b><a href="https://github.com/ATOMScience-org/AMPL#hyperparameter-optimization">here</a></b>

.. |params| raw:: html

   <b><a href="https://github.com/ATOMScience-org/AMPL/blob/master/atomsci/ddm/docs/PARAMETERS.md">here</a></b>

.. |xgboost| raw:: html

   <b><a href="https://en.wikipedia.org/wiki/XGBoost">XGBoost</a></b>

.. |ecfp| raw:: html

   <b><a href="https://pubs.acs.org/doi/10.1021/ci100050t">ECFP</a></b>