####################################################
07 Compare Models to Select the Best Hyperparameters
####################################################

*Published: June, 2024, ATOM DDM Team*

------------

This tutorial will review
`AMPL <https://github.com/ATOMScience-org/AMPL>`_ functions for
visualizing the results of a hyperparameter search in order to find the
optimal hyperparameters for your model.

After performing a hyperparameter search, it is prudent to examine each
hyperparameter in order to determine the best combination before
training a production model with all of the data. Additionally, it is
good to explore multiple performance metrics and visualize the
predictions instead of relying solely on metrics.

For the purposes of this tutorial, we simply ran **Tutorial 6,
"Hyperparameter Optimization"**, with different parameters such as those
outlined
`here <https://github.com/ATOMScience-org/AMPL#hyperparameter-optimization>`_
to get enough models for comparison. Specifically, we created additional
**NN** and `XGBoost <https://en.wikipedia.org/wiki/XGBoost>`_
models as well as using **fingerprint** and **scaffold splits**. If you
don't want to run that many models, you can use the ``result_df`` saved
here: ``dataset/SLC6A3_models/07_example_pred_df.csv``.

In this tutorial, we will focus on these functions:

-  `plot_train_valid_test_scores <https://ampl.readthedocs.io/en/latest/pipeline.html#pipeline.hyper_perf_plots.plot_train_valid_test_scores>`_
-  `plot_split_perf <https://ampl.readthedocs.io/en/latest/pipeline.html#pipeline.hyper_perf_plots.plot_split_perf>`_
-  `plot_hyper_perf <https://ampl.readthedocs.io/en/latest/pipeline.html#pipeline.hyper_perf_plots.plot_hyper_perf>`_
-  `get_score_types <https://ampl.readthedocs.io/en/latest/pipeline.html#pipeline.hyper_perf_plots.get_score_types>`_
-  `plot_xg_perf <https://ampl.readthedocs.io/en/latest/pipeline.html#pipeline.hyper_perf_plots.plot_xg_perf>`_
-  `plot_pred_vs_actual_from_file <https://ampl.readthedocs.io/en/latest/pipeline.html#pipeline.perf_plots.plot_pred_vs_actual_from_file>`_

Import Packages
***************

.. code:: ipython3

    from atomsci.ddm.pipeline import compare_models as cm
    from atomsci.ddm.pipeline import hyper_perf_plots as hpp
    from atomsci.ddm.pipeline import perf_plots as pp
    import pandas as pd
    pd.set_option('display.max_columns', None)
    
    # ignore warnings in tutorials
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)



Get Model Results and Filter
****************************

First we pull the results of the hyperparameter search into a dataframe.
In **Tutorial 6, "Hyperparameter Optimization"**, we used
``get_filesystem_perf_results()`` which packs hyperparameters in a dict
in the column ``model_parameters_dict``. Here we use the individual
hyperparameter columns to create visualizations.

The ``result_df`` used here as an example is the result of calling
``get_filesystem_perf_results()`` once after training several hundred
models with different parameters. These models were all saved in a
single folder, but the function works iteratively so it can search an
entire directory if a parent folder is passed.

.. code:: ipython3

    # # call the function yourself
    # result_df=cm.get_filesystem_perf_results(result_dir='dataset/SLC6A3_models/', pred_type='regression')
    
    # use example results
    result_df=pd.read_csv('dataset/SLC6A3_models/07_example_pred_df.csv', index_col=0)
    
    result_df=result_df.sort_values('best_valid_r2_score', ascending=False)
    print(result_df.shape)
    result_df.head(2)


.. list-table:: 
   :header-rows: 1
   :class: tight-table 
 
   * - 
     - model_uuid
     - model_path
     - ampl_version
     - model_type
     - dataset_key
     - features
     - ...
   * - 310
     - b24a2887-8eca-43e2-8fc2-3642189d2c94
     - dataset/SLC6A3_models/SLC6A3_Ki_curated_model_...
     - 1.6.0
     - NN
     - /home/apaulson/repos/AMPL_umbrella/AMPL/atomsc...
     - ecfp
     - ...
   * - 306
     - 9b6c9332-15f3-4f96-9579-bf407d0b69a8
     - dataset/SLC6A3_models/SLC6A3_Ki_curated_model_...
     - 1.6.0
     - RF
     - /home/apaulson/repos/AMPL_umbrella/AMPL/atomsc...
     - ecfp
     - ...


We can look at a brief count of models for important parameters by
creating a pivot table. Here we can see `ECFP
fingerprints <https://pubs.acs.org/doi/10.1021/ci100050t>`_ and
`RDKit <https://github.com/rdkit/rdkit>`_ features and
**fingerprint** and **scaffold splitters** were used for each model
type.

.. code:: ipython3

    # model counts
    model_counts=pd.DataFrame(result_df.groupby(['features','splitter','model_type'])['model_uuid'].count()).reset_index()
    model_counts=model_counts.pivot(index='model_type',columns=['splitter','features',], values='model_uuid')
    model_counts



.. list-table:: 
   :header-rows: 3
   :class: tight-table 
 
   * - splitter
     - fingerprint
     - scaffold
     - fingerprint
     - scaffold
   * - features
     - ecfp
     - ecfp
     - rdkit_raw
     - rdkit_raw
   * - model_type
     -      
     -      
     -      
     -      
   * - NN
     - 26
     - 29
     - 25
     - 96
   * - RF
     - 30
     - 30
     - 30
     - 32
   * - xgboost
     - 47
     - 26
     - 20
     - 76



Often, certain random combinations of hyperparameters result in terribly
performing models. Here we will filter those out so they don't affect
the visualization by only keeping models with a validation ``r2_score``
of 0.1 or greater.


.. code:: ipython3

    result_df.best_valid_r2_score.describe()


.. code:: ipython3

    # filter out objectively bad performing models
    result_df=result_df[result_df.best_valid_r2_score>0.1]
    result_df.shape


.. code:: ipython3

    result_df.best_valid_r2_score.describe()


After filtering out models with extremely poor metrics, we can see that
some combinations don't work at all, and are completely filtered from
the set. For example, decision tree based models using
`RDKit <https://github.com/rdkit/rdkit>`_ or
`ECFP <https://pubs.acs.org/doi/10.1021/ci100050t>`_ features work
very poorly to predict on fingerprint-split models.

.. code:: ipython3

    #  model counts
    model_counts=pd.DataFrame(result_df.groupby(['features','splitter','model_type'])['model_uuid'].count()).reset_index()
    model_counts=model_counts.pivot(index='model_type',columns=['splitter','features',], values='model_uuid')
    model_counts



.. list-table:: 
   :header-rows: 3
   :class: tight-table 
 
   * - splitter
     - fingerprint
     - scaffold
     - fingerprint
     - scaffold
   * - features
     - ecfp
     - ecfp
     - rdkit_raw
     - rdkit_raw
   * - model_type
     -      
     -      
     -      
     -      
   * - NN
     - 8.0		
     - 23.0
     - 11.0
     - 86.0
   * - RF
     - NaN
     - 30.0
     - NaN
     - 32.0
   * - xgboost
     - 3.0			
     - 21.0
     - NaN
     - 50.0


Visualize Hyperparameters
*************************

There are several plotting functions in the ``hyper_perf_plots`` module
that help visualize the different combinations of features for each type
of model.

Examine overall scores
----------------------

``plot_train_valid_test_scores()`` gives a quick snapshot of your
overall model performance. You can see if you overfitted and get a sense
of whether your partitions are a good representation of future
performance. Because the splitter can have a drastic effect on model
performance, these plots are also separated by split type.

Here we see a fairly typical pattern where the training set metrics are
higher than validation and test partitions. It is good to see that the
validation and test scores are similar across many models, indicating
that the models are not overfitting to the validation data. For
fingerprint splits, we see a more worrying trend where the validation
and test set metrics are not as consistent, suggesting that the split is
not helping the model to generalize.

.. code:: ipython3

    hpp.plot_train_valid_test_scores(result_df, prediction_type='regression')



.. image:: ../_static/img/07_compare_models_files/07_compare_models_15_0.png


Examine Splits
--------------

``plot_split_perf()`` plots the performance of each split type,
separated by feature type, for each performance metric.

We can see that **fingerprint splits** perform much worse than
**scaffold splits** for this dataset, and but
`RDKit <https://github.com/rdkit/rdkit>`_ and
`ECFP <https://pubs.acs.org/doi/10.1021/ci100050t>`_ features
perform differently.
`ECFP <https://pubs.acs.org/doi/10.1021/ci100050t>`_ features work
better for scaffold splits while
`RDKit <https://github.com/rdkit/rdkit>`_ features work better for
**fingerprint splits**. Recalling the filtering from above, we know that
`RDKit <https://github.com/rdkit/rdkit>`_ features for
**fingerprint splits** are only represented by **NN** models, which may
skew these results.

.. code:: ipython3

    hpp.plot_split_perf(result_df, subset='valid')



.. image:: ../_static/img/07_compare_models_files/07_compare_models_17_0.png


General Model Features
======================

We also want to understand general hyperparameters like model type and
feature type and their effect on performance. We can use
``plot_hyper_perf()`` with ``model_type='general'`` as a shortcut to
visualize these.

We can see that **random forests** or **neural networks** perform the
best while `ECFP <https://pubs.acs.org/doi/10.1021/ci100050t>`_
features perform best while
`ECFP <https://pubs.acs.org/doi/10.1021/ci100050t>`_ features
perform better than **RDKit**. Additionally, the **random forest**
models are very consistent while there is more variability in the **NN**
and **XGBoost** model performance.

.. code:: ipython3

    hpp.plot_hyper_perf(result_df, model_type='general')



.. image:: ../_static/img/07_compare_models_files/07_compare_models_19_0.png


RF-specific Hyperparameters
--------------------------

We can also use ``plot_hyper_perf()`` to visualize model-specific
hyperparameters. In this case we examine random forest models because
they generally perform the best for this dataset.

Here, we can see two distinct sets of ``valid_r2_scores`` (probably from
**fingerprint** vs **scaffold split** models), but both sets show
similar trends. For ``rf_estimators`` it looks like 100-150 trees is
optimal, while ``rf_max_depth`` does worse below ~15 and improves slowly
after that. ``rf_max_features`` doesn't show a clear trend except that
below 50 might result in worse models.

.. code:: ipython3

    hpp.plot_hyper_perf(result_df, model_type='RF', subset='valid', scoretype='r2_score')



.. image:: ../_static/img/07_compare_models_files/07_compare_models_21_0.png


We can quickly get a list of scores to plot with ``get_score_types()``
and create the same plots with different metrics.

.. code:: ipython3

    hpp.get_score_types()


.. code:: ipython3

    hpp.plot_hyper_perf(result_df, model_type='RF', subset='valid', scoretype='mae_score')



.. image:: ../_static/img/07_compare_models_files/07_compare_models_24_0.png


NN Visualization
================

When visualizing hyperparameters of NN models in this case, it is
slightly hard to see important trends because there is a large variance
in their model performance. To avoid this, we use ``plot_hyper_perf()``
with a subsetted dataframe to look at a single combination of splitter
and features.

.. list-table:: 
   :header-rows: 1
   :class: tight-table 
   
   * - Plot Features
     - Description
   * - `avg_dropout`
     - The average of dropout proportions across all layers of the model. This parameter can affect the generalizability and overfitting of the model and usually dropout of 0.1 or higher is best.
   * - `learning_rate`
     - The learning rate during training. Generally, learning rates that are ~10e-3 do best.
   * - `num_weights`
     - The product of layer sizes plus number of nodes in first layer, a rough estimate of total model size/complexity. This parameter should be minimized by selecting the smallest layer sizes possible that still maximize the preferred metric
   * - `num_layers`
     - The number of layers in the NN, another marker of complexity. This should also be minimized.
   * - `best_epoch`
     - Which epoch had the highest performance metric during training. This can indicate problematic training if the best_epochs are very small.
   * - `max_epochs`
     - The max number of epochs the model was allowed to train (although "early stopping" may have occurred). If the max_epochs is too small you may underfit your model. This could be shown by all of your best_epochs being at max_epoch.

.. code:: ipython3

    subsetted=result_df[result_df.splitter=='scaffold']
    subsetted=subsetted[subsetted.features=='rdkit_raw']
    
    hpp.plot_hyper_perf(subsetted, model_type='NN')



.. image:: ../_static/img/07_compare_models_files/07_compare_models_26_0.png


XGBoost Visualization
=====================

Using ``plot_xg_perf()``, we can simultaneously visualize the two most
important parameters for
`XGBoost <https://en.wikipedia.org/wiki/XGBoost>`_ models - the
learning rate and gamma. We can see that ``xgb_learning_rate`` should be
between 0 and 0.45, after which the performance starts to deteriorate.
There's no clear trend for ``xgb_gamma``. We can additionally use
``plot_hyper_perf()`` to visualize more
`XGBoost <https://en.wikipedia.org/wiki/XGBoost>`_ parameters, but
this is not shown here.

.. code:: ipython3

    # hpp.plot_hyper_perf(result_df, model_type='xgboost')

    hpp.plot_xg_perf(result_df)



.. image:: ../_static/img/07_compare_models_files/07_compare_models_29_0.png


Evaluation of a Single Model
****************************

After calling ``compare_models.get_filesystem_perf_results()``, the
dataframe can be sorted according to the score you care about. The
column ``model_parameters_dict`` contains hyperparameters used for the
best model. We can visualize this model using
``perf_plots.plot_pred_vs_actual_from_file()``.

.. note::
    
    *not all scores should be maximized. For example,
    ``mae_score`` or ``rms_score`` should be minimized instead.*

.. code:: ipython3

    winnertype='best_valid_r2_score'
    
    # result_df=cm.get_filesystem_perf_results(result_dir='dataset/SLC6A3_models/', pred_type='regression')
    
    result_df=pd.read_csv('dataset/SLC6A3_models/07_example_pred_df.csv', index_col=0)
    result_df=result_df.sort_values(winnertype, ascending=False)

We can examine important parameters of the top model directly from the
``result_df``.

We see that through hyperparameter optimization, we have increased our
``best_valid_r2_score`` to 0.56, as compared to our baseline model
``valid_r2_score`` of 0.50011 (from **Tutorial 4, "Train a Simple
Regression Model"**).

.. code:: ipython3

    result_df.iloc[0][['features','splitter','best_valid_r2_score']]


.. code:: ipython3

    result_df.iloc[0].model_parameters_dict


.. code:: ipython3

    result_df.iloc[0].model_path


Here we use ``plot_pred_vs_actual_from_file()`` to visualize the
prediction accuracy for the train, validation and test sets. 

.. note::

    *For the purposes of this tutorial, the following models have been
    altered to work on every file system. In general, this function is meant
    for quick analysis of models you've trained on your own machine. To use
    an external model and predict on external data, see **Tutorial 5,
    "Application of a Trained Model"**.*

.. code:: ipython3

    # plot best model, an NN
    import importlib
    importlib.reload(pp)
    model_path=result_df.iloc[0].model_path
    pp.plot_pred_vs_actual_from_file(model_path)


.. image:: ../_static/img/07_compare_models_files/07_compare_models_38_2.png


This NN model looks like it isn't very good at predicting things with
:math:`pKi` < 4.5. Additionally, there is a set of data at
:math:`pKi`\ =5 (this data is censored and all we know is that the
compounds have a :math:`pKi` < 5 because higher concentrations of drug
were not tested). This data is poorly predicted by the NN model. 

.. note::

    *Be wary of selecting models only based on their performance
    metrics! As we can see, this NN has problems even though the r2\_score
    is fairly high.*

.. code:: ipython3

    # plot best RF model
    model_type='RF'
    model_path=result_df[result_df.model_type==model_type].iloc[0].model_path
    pp.plot_pred_vs_actual_from_file(model_path)
    print('\nBest valid r2 score: ',result_df[result_df.model_type==model_type].iloc[0].best_valid_r2_score)
    print('\nModel Parameters: ',result_df[result_df.model_type==model_type].iloc[0].model_parameters_dict,'\n')



.. image:: ../_static/img/07_compare_models_files/07_compare_models_40_2.png


This RF model looks like it did better at training than the best NN
model, even though its performance validation score is slightly lower.
The low :math:`pKi` values are learned more accurately in the training
set, and the censored data at :math:`pKi`\ =5 is also predicted more
accurately.

.. code:: ipython3

    # plot best xgboost model
    model_type='xgboost'
    model_path=result_df[result_df.model_type==model_type].iloc[0].model_path
    pp.plot_pred_vs_actual_from_file(model_path)
    print('\nBest valid r2 score: ',result_df[result_df.model_type==model_type].iloc[0].best_valid_r2_score)
    print('\nModel Parameters: ',result_df[result_df.model_type==model_type].iloc[0].model_parameters_dict,'\n')



.. image:: ../_static/img/07_compare_models_files/07_compare_models_42_2.png


This `XGBoost <https://en.wikipedia.org/wiki/XGBoost>`_ model
learns the low :math:`pKi` values better but still suffers from problems
with predicting the censored data.

Moving forward, we would select the ``RF`` model as the best performer.

In **Tutorial 8, "Train a Production Model"**, we will use the
best-performing parameters to create a production model for the entire
dataset.

If you have specific feedback about a tutorial, please complete the `AMPL Tutorial Evaluation <https://forms.gle/pa9sHj4MHbS5zG7A6>`_.
