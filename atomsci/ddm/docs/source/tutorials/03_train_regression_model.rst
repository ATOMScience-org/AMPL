##################################
03 Train a Simple Regression Model
##################################

*Published: June, 2024, ATOM DDM Team*

------------


The process of training a machine learning (ML) model can be thought of
as fitting a highly parameterized function to map inputs to outputs. An
ML algorithm needs to train numerous examples of input and output pairs
to accurately map an input to an output, i. e., make a prediction. After
training, the result is referred to as a trained ML model or an
artifact.

This tutorial will detail how we can use
`AMPL <https://github.com/ATOMScience-org/AMPL>`_ tools to train a
regression model to predict how much a compound will inhibit the
`SLC6A3 <https://www.ebi.ac.uk/chembl/target_report_card/CHEMBL238/>`_
protein as measured by :math:`pK_i`. We will train a random forest model
using the following inputs:

1. The curated
   `SLC6A3 <https://www.ebi.ac.uk/chembl/target_report_card/CHEMBL238>`_
   dataset from **Tutorial 1, "Data Curation"**.
2. The split file generated in **Tutorial 2, "Splitting Datasets for
   Validation and Testing"**.
3. `RDKit <https://github.com/rdkit/rdkit>`_ features calculated by
   the `AMPL <https://github.com/ATOMScience-org/AMPL>`_ pipeline.

The tutorial will present the following functions and classes:

-  `ModelPipeline <https://ampl.readthedocs.io/en/latest/pipeline.html#module-pipeline.model_pipeline>`_
-  `parameter_parser.wrapper <https://ampl.readthedocs.io/en/latest/pipeline.html#pipeline.parameter_parser.wrapper>`_
-  `compare_models.get_filesystem_perf_results <https://ampl.readthedocs.io/en/latest/pipeline.html#pipeline.compare_models.get_filesystem_perf_results>`_

We will explain the use of descriptors, how to evaluate model
performance, and where the model is saved as a .tar.gz file.

.. note::   
    
    *Training a random forest model and splitting the dataset
    are non-deterministic. You will obtain a slightly different random
    forest model by running this tutorial each time.*

Model Training (Using Previously Split Data)
********************************************

In our first example, we train a model using a curated dataset (as
described in **Tutorial 1, “Data Curation”**) that was already split
using the procedure in **Tutorial 2, “Splitting Datasets for Validation
and Testing”**. To use an existing split file, we specify its
``split_uuid`` in the model parameters and set the ``previously_split``
parameter to True. In the example code, we set ``split_uuid`` to point
to a split file provided with AMPL, in case you’re running this tutorial
without having previously done **Tutorial 2, "Splitting Datasets for
Validation and Testing"**.

Here, we will use
``"split_uuid": "c35aeaab-910c-4dcf-8f9f-04b55179aa1a"`` which is saved
in ``dataset/`` as a convenience for these tutorials.

`AMPL <https://github.com/ATOMScience-org/AMPL>`_ provides an
extensive featurization module that can generate a variety of molecular
feature types, given
`SMILES <https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system>`_
strings as input. For demonstration purposes, we choose to use
`RDKit <https://github.com/rdkit/rdkit>`_ features in this
tutorial.

When the featurized dataset is not previously saved for SLC6A3\_Ki,
`AMPL <https://github.com/ATOMScience-org/AMPL>`_ will create a
featurized dataset and save it in a folder called ``scaled_descriptors``
as a csv file e.g.
``dataset/scaled_descriptors/SLC6A3_Ki_curated_with_rdkit_raw_descriptors.csv``.

After training, `AMPL <https://github.com/ATOMScience-org/AMPL>`_
saves the model and all of its parameters as a tarball in the directory
given by ``result_dir``.

.. code:: ipython3

    # importing relevant libraries
    import pandas as pd
    pd.set_option('display.max_columns', None)
    from atomsci.ddm.pipeline import model_pipeline as mp
    from atomsci.ddm.pipeline import parameter_parser as parse
    
    # Set up
    dataset_file = 'dataset/SLC6A3_Ki_curated.csv'
    odir='dataset/SLC6A3_models/'
    
    response_col = "avg_pKi"
    compound_id = "compound_id"
    smiles_col = "base_rdkit_smiles"
    split_uuid = "c35aeaab-910c-4dcf-8f9f-04b55179aa1a"
    
    params = {
            "prediction_type": "regression",
            "dataset_key": dataset_file,
            "id_col": compound_id,
            "smiles_col": smiles_col,
            "response_cols": response_col,
            "previously_split": "True",
            "split_uuid" : split_uuid,
            "split_only": "False",
            "featurizer": "computed_descriptors",
            "descriptor_type" : "rdkit_raw",
            "model_type": "RF",
            "verbose": "True",
            "transformers": "True",
            "rerun": "False",
            "result_dir": odir
        }
    
    ampl_param = parse.wrapper(params)
    pl = mp.ModelPipeline(ampl_param)
    pl.train_model()


Model Training (Split Data and Train)
*************************************

`AMPL <https://github.com/ATOMScience-org/AMPL>`_ also provides an option to split a dataset and train a model in one
step, by setting the ``previously_split`` parameter to False and
omitting the ``split_uuid`` parameter.
`AMPL <https://github.com/ATOMScience-org/AMPL>`_ splits the data
by the type of split specified in the splitter parameter, scaffold in
this example, and writes the split file in
``dataset/SLC6A3_Ki_curated_train_valid_test_scaffold_{split_uuid}.csv``

Although it's convenient, it is not a good idea to use the one-step
option if you intend to train multiple models with different parameters
on the same dataset and compare their performance. If you do, you will
end up with different splits for each model, and won't be able to tell
if the differences in performance are due to the parameter settings or
to the random variations between splits.

.. code:: ipython3

    params = {
            "prediction_type": "regression",
            "dataset_key": dataset_file,
            "id_col": compound_id,
            "smiles_col": smiles_col,
            "response_cols": response_col,
        
            "previously_split": "False",
            "split_only": "False",
            "splitter": "scaffold",
            "split_valid_frac": "0.15",
            "split_test_frac": "0.15",
        
            "featurizer": "computed_descriptors",
            "descriptor_type" : "rdkit_raw",
            "model_type": "RF",
            "transformers": "True",
            "rerun": "False",
            "result_dir": odir
        }
    
    ampl_param = parse.wrapper(params)
    pl = mp.ModelPipeline(ampl_param)
    pl.train_model()

Performance of the Model
************************

We evaluate model performance by measuring how accurate models are on
validation and test sets. The validation set is used while optimizing
the model and choosing the best parameter settings. Finally, we use the
model's performance on the test set to judge the model.

`AMPL <https://github.com/ATOMScience-org/AMPL>`_ has several
popular metrics to evaulate regression models; **Mean Absolute Error
(MAE)**, **Root Mean Squared Error (RMSE)** and :math:`R^2` (R-Squared).
In our tutorials, we will use :math:`R^2` metric to compare our models.
The best model will have the highest :math:`R^2` score.

.. code:: ipython3

    # Model Performance
    from atomsci.ddm.pipeline import compare_models as cm
    perf_df = cm.get_filesystem_perf_results(odir, pred_type='regression')


.. parsed-literal::

    Found data for 2 models under dataset/SLC6A3_models/


The ``perf_df`` dataframe has details about the ``model_uuid``,
``model_path``, ``ampl_version``, ``model_type``, ``features``,
``splitter``\ and the results for popular metrics that help evaluate the
performance. Let us view the contents of the ``perf_df`` dataframe.

.. code:: ipython3

    # save perf_df
    import os
    perf_df.to_csv(os.path.join(odir, 'perf_df.csv'))

.. code:: ipython3

    # View the perf_df dataframe
    
    # show most useful columns
    perf_df[['model_uuid', 'split_uuid', 'best_train_r2_score', 'best_valid_r2_score', 'best_test_r2_score']]




.. list-table:: 
   :header-rows: 1
   :class: tight-table 
 
   * - 
     - model_uuid
     - split_uuid
     - best_train_r2_score
     - best_valid_r2_score
     - best_test_r2_score
   * - 0
     - 9ff5a924-ef49-407c-a4d4-868a1288a67e
     - c35aeaab-910c-4dcf-8f9f-04b55179aa1a
     - 0.949835
     - 0.500110
     - 0.426594
   * - 1
     - f69409b0-33ce-404f-b1e5-0e9f5128ebc7
     - f6351696-363f-411a-8720-4892bc4f700e
     - 0.949919
     - 0.472619
     - 0.436174




Finding the Top Performing Model
********************************

To pick the top performing model, we sort the performance table by
``best_valid_r2_score`` in descending order and examine the top row.

.. code:: ipython3

    # Top performing model
    top_model=perf_df.sort_values(by="best_valid_r2_score", ascending=False).iloc[0]
    top_model




.. parsed-literal::

    model_uuid                               9ff5a924-ef49-407c-a4d4-868a1288a67e
    model_path                  dataset/SLC6A3_models/SLC6A3_Ki_curated_model_...
    ampl_version                                                            1.6.1
    model_type                                                                 RF
    dataset_key                 /Users/rwilfong/Downloads/2024_LLNL/fork_ampl/...
    features                                                            rdkit_raw
    splitter                                                             scaffold
    split_strategy                                               train_valid_test
    split_uuid                               c35aeaab-910c-4dcf-8f9f-04b55179aa1a
    model_score_type                                                           r2
    feature_transform_type                                          normalization
    weight_transform_type                                                    None
    model_choice_score                                                    0.50011
    best_train_r2_score                                                  0.949835
    best_train_rms_score                                                  0.27884
    best_train_mae_score                                                 0.198072
    best_train_num_compounds                                                 1273
    best_valid_r2_score                                                   0.50011
    best_valid_rms_score                                                 0.854443
    best_valid_mae_score                                                 0.700053
    best_valid_num_compounds                                                  273
    best_test_r2_score                                                   0.426594
    best_test_rms_score                                                   0.92241
    best_test_mae_score                                                  0.746781
    best_test_num_compounds                                                   273
    rf_estimators                                                             500
    rf_max_features                                                            32
    rf_max_depth                                                             None
    max_epochs                                                                NaN
    best_epoch                                                                NaN
    learning_rate                                                             NaN
    layer_sizes                                                               NaN
    dropouts                                                                  NaN
    xgb_gamma                                                                 NaN
    xgb_learning_rate                                                         NaN
    xgb_max_depth                                                             NaN
    xgb_colsample_bytree                                                      NaN
    xgb_subsample                                                             NaN
    xgb_n_estimators                                                          NaN
    xgb_min_child_weight                                                      NaN
    model_parameters_dict       {"rf_estimators": 500, "rf_max_depth": null, "...
    feat_parameters_dict                                                       {}
    Name: 0, dtype: object



You can find the path to the .tar.gz file ("tarball") where the top
performing model is saved by examining ``top_model.model_path``. You
will need this path to run predictions with the model at a later time.

.. code:: ipython3

    # Top performing model path
    top_model.model_path




.. parsed-literal::

    'dataset/SLC6A3_models/SLC6A3_Ki_curated_model_9ff5a924-ef49-407c-a4d4-868a1288a67e.tar.gz'



In **Tutorial 4 , "Application of a Trained Model"**, we will learn how
to use a selected model to make predictions and evaluate those
predictions

If you have specific feedback about a tutorial, please complete the
`AMPL Tutorial Evaluation <https://forms.gle/pa9sHj4MHbS5zG7A6>`_.
