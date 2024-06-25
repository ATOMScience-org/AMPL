Hyperparameter Optimization
===========================

Hyperparameters dictate the parameters of the training process and the
architecture of the model itself. For example, the number of random
trees is a hyperparameter for a **random forest**. In contrast, a
learned parameter for a **random forest** is the set of features that is
contained in a single node (in a single tree) and the cutoff values for
each of those features that determines how the data is split at that
node. A full discussion of hyperparameter optimization can be found on
**`Wikipedia <https://en.wikipedia.org/wiki/Hyperparameter_optimization>`__**.

The choice of hyperparameters strongly influences model performance, so
it is important to be able to optimize them as well.
**`AMPL <https://github.com/ATOMScience-org/AMPL>`__** offers a variety
of hyperparameter optimization methods including random sampling, grid
search, and Bayesian optimization. Please refer to the parameter
documentation
**`page <https://github.com/ATOMScience-org/AMPL#hyperparameter-optimization>`__**
for further information.

In this tutorial we demonstrate the following: - Build a parameter
dictionary to perform a hyperparameter search for a **random forest**
using Bayesian optimization. - Perform the optimization process. -
Review the results

We will use these **`AMPL <https://github.com/ATOMScience-org/AMPL>`__**
functions: -
`parse\_params <https://ampl.readthedocs.io/en/latest/utils.html#utils.hyperparam_search_wrapper.parse_params>`__
-
`build\_search <https://ampl.readthedocs.io/en/latest/utils.html#utils.hyperparam_search_wrapper.build_search>`__
-
`run\_search <https://ampl.readthedocs.io/en/latest/utils.html#utils.hyperparam_search_wrapper.HyperOptSearch.run_search>`__
-
`get\_filesystem\_perf\_results <https://ampl.readthedocs.io/en/latest/pipeline.html#pipeline.compare_models.get_filesystem_perf_results>`__

The first three functions in the above list come from the
``hyperparameter_search_wrapper`` module.

Set Up Directories
------------------

Here we set up a few important variables corresponding to required
directories and specific features for the **hyperparameter optimization
(HPO)** process. Then, we ensure that the directories are created before
saving models into them.

+------+------+
| Vari | Desc |
| able | ript |
|      | ion  |
+======+======+
| ``da | The  |
| tase | rela |
| t_ke | tive |
| y``  | path |
|      | to   |
|      | the  |
|      | data |
|      | set  |
|      | you  |
|      | want |
|      | to   |
|      | use  |
|      | for  |
|      | HPO  |
+------+------+
| ``de | The  |
| scri | type |
| ptor | of   |
| _typ | feat |
| e``  | ures |
|      | you  |
|      | want |
|      | to   |
|      | use  |
|      | duri |
|      | ng   |
|      | HPO  |
+------+------+
| ``mo | The  |
| del_ | dire |
| dir` | ctor |
| `    | y    |
|      | wher |
|      | e    |
|      | you  |
|      | want |
|      | to   |
|      | save |
|      | all  |
|      | of   |
|      | the  |
|      | mode |
|      | ls   |
+------+------+
| ``be | For  |
| st_m | Baye |
| odel | sian |
| _dir | opti |
| ``   | miza |
|      | tion |
|      | ,    |
|      | the  |
|      | winn |
|      | ing  |
|      | mode |
|      | l    |
|      | is   |
|      | save |
|      | d    |
|      | in   |
|      | this |
|      | sepa |
|      | rate |
|      | fold |
|      | er   |
+------+------+
| ``sp | The  |
| lit_ | pres |
| uuid | aved |
| ``   | spli |
|      | t    |
|      | uuid |
|      | from |
|      | **Tu |
|      | tori |
|      | al   |
|      | 2,   |
|      | "Spl |
|      | itti |
|      | ng   |
|      | Data |
|      | sets |
|      | for  |
|      | Vali |
|      | dati |
|      | on   |
|      | and  |
|      | Test |
|      | ing" |
|      | **   |
+------+------+

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
-----------------------------

+------+------+
| Para | Desc |
| mete | ript |
| r    | ion  |
+======+======+
| ``'h | This |
| yper | sett |
| para | ing  |
| m':' | indi |
| True | cate |
| '``  | s    |
|      | that |
|      | we   |
|      | are  |
|      | perf |
|      | ormi |
|      | ng   |
|      | a    |
|      | hype |
|      | rpar |
|      | amet |
|      | er   |
|      | sear |
|      | ch   |
|      | inst |
|      | ead  |
|      | of   |
|      | just |
|      | trai |
|      | ning |
|      | one  |
|      | mode |
|      | l.   |
+------+------+
| ``'p | This |
| revi | tell |
| ousl | s    |
| y_fe | **`A |
| atur | MPL  |
| ized | <htt |
| ':'T | ps:/ |
| rue' | /git |
| ``   | hub. |
|      | com/ |
|      | ATOM |
|      | Scie |
|      | nce- |
|      | org/ |
|      | AMPL |
|      | >`__ |
|      | **   |
|      | to   |
|      | sear |
|      | ch   |
|      | for  |
|      | prev |
|      | ious |
|      | ly   |
|      | gene |
|      | rate |
|      | d    |
|      | feat |
|      | ures |
|      | in   |
|      | ``.. |
|      | /dat |
|      | aset |
|      | /sca |
|      | led_ |
|      | desc |
|      | ript |
|      | ors` |
|      | `    |
|      | inst |
|      | ead  |
|      | of   |
|      | rege |
|      | nera |
|      | ting |
|      | them |
|      | on   |
|      | the  |
|      | fly. |
+------+------+
| ``'s | This |
| earc | spec |
| h_ty | ifie |
| pe': | s    |
| 'hyp | the  |
| erop | hype |
| t'`` | rpar |
|      | amet |
|      | er   |
|      | sear |
|      | ch   |
|      | meth |
|      | od.  |
|      | Othe |
|      | r    |
|      | opti |
|      | ons  |
|      | incl |
|      | ude  |
|      | ``gr |
|      | id`` |
|      | ,    |
|      | ``ra |
|      | ndom |
|      | ``,  |
|      | and  |
|      | ``ge |
|      | omet |
|      | ric` |
|      | `.   |
|      | Spec |
|      | ific |
|      | atio |
|      | ns   |
|      | for  |
|      | each |
|      | hype |
|      | rpar |
|      | amet |
|      | er   |
|      | sear |
|      | ch   |
|      | meth |
|      | od   |
|      | is   |
|      | diff |
|      | eren |
|      | t,   |
|      | plea |
|      | se   |
|      | refe |
|      | r    |
|      | to   |
|      | the  |
|      | full |
|      | docu |
|      | ment |
|      | atio |
|      | n.   |
|      | Here |
|      | we   |
|      | are  |
|      | usin |
|      | g    |
|      | the  |
|      | Baye |
|      | sian |
|      | opti |
|      | miza |
|      | tion |
|      | meth |
|      | od.  |
+------+------+
| ``'m | This |
| odel | mean |
| _typ | s    |
| e':' | **`A |
| RF\| | MPL  |
| 10'` | <htt |
| `    | ps:/ |
|      | /git |
|      | hub. |
|      | com/ |
|      | ATOM |
|      | Scie |
|      | nce- |
|      | org/ |
|      | AMPL |
|      | >`__ |
|      | **   |
|      | will |
|      | try  |
|      | 10   |
|      | time |
|      | s    |
|      | to   |
|      | find |
|      | the  |
|      | best |
|      | set  |
|      | of   |
|      | hype |
|      | rpar |
|      | amet |
|      | ers  |
|      | usin |
|      | g    |
|      | **ra |
|      | ndom |
|      | fore |
|      | sts* |
|      | *.   |
|      | In   |
|      | prac |
|      | tice |
|      | ,    |
|      | this |
|      | para |
|      | mete |
|      | r    |
|      | coul |
|      | d    |
|      | be   |
|      | set  |
|      | to   |
|      | 100  |
|      | or   |
|      | more |
|      | .    |
+------+------+
| ``'r | The  |
| fe': | Baye |
| 'uni | sian |
| form | opti |
| int\ | mize |
| |8,5 | r    |
| 12'` | will |
| `    | unif |
|      | orml |
|      | y    |
|      | sear |
|      | ch   |
|      | betw |
|      | een  |
|      | 8    |
|      | and  |
|      | 512  |
|      | for  |
|      | the  |
|      | best |
|      | numb |
|      | er   |
|      | of   |
|      | rand |
|      | om   |
|      | fore |
|      | st   |
|      | esti |
|      | mato |
|      | rs.  |
|      | Simi |
|      | larl |
|      | y    |
|      | ``rf |
|      | d``  |
|      | stan |
|      | ds   |
|      | for  |
|      | **ra |
|      | ndom |
|      | fore |
|      | st   |
|      | dept |
|      | h**  |
|      | and  |
|      | ``rf |
|      | f``  |
|      | stan |
|      | ds   |
|      | for  |
|      | **ra |
|      | ndom |
|      | fore |
|      | st   |
|      | feat |
|      | ures |
|      | **.  |
+------+------+
| ``re | Now  |
| sult | expe |
| _dir | cts  |
| ``   | two  |
|      | para |
|      | mete |
|      | rs.  |
|      | The  |
|      | firs |
|      | t    |
|      | dire |
|      | ctor |
|      | y    |
|      | will |
|      | cont |
|      | ain  |
|      | the  |
|      | best |
|      | trai |
|      | ned  |
|      | mode |
|      | ls   |
|      | whil |
|      | e    |
|      | the  |
|      | seco |
|      | nd   |
|      | dire |
|      | ctor |
|      | y    |
|      | will |
|      | cont |
|      | ain  |
|      | all  |
|      | mode |
|      | ls   |
|      | trai |
|      | ned  |
|      | in   |
|      | the  |
|      | sear |
|      | ch.  |
+------+------+

Regression models are optimized to maximize the :math:`R^2` and
classification models are optimized using area under the receiver
operating characteristic curve. A full list of parameters can be found
on our
**`github <https://github.com/ATOMScience-org/AMPL/blob/master/atomsci/ddm/docs/PARAMETERS.md>`__**.

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
-------------------------

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




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>model_uuid</th>
          <th>model_parameters_dict</th>
          <th>best_valid_r2_score</th>
          <th>best_test_r2_score</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>4</th>
          <td>dbd1d89c-05f5-4224-bce4-7dbeafaba313</td>
          <td>{"rf_estimators": 262, "rf_max_depth": 16, "rf...</td>
          <td>0.488461</td>
          <td>0.424234</td>
        </tr>
        <tr>
          <th>8</th>
          <td>601ae89f-a8bb-4da2-b7a7-b434a2bdcbbe</td>
          <td>{"rf_estimators": 190, "rf_max_depth": 15, "rf...</td>
          <td>0.483822</td>
          <td>0.448591</td>
        </tr>
        <tr>
          <th>9</th>
          <td>0967e5ea-64a1-4509-80da-176bd8773775</td>
          <td>{"rf_estimators": 146, "rf_max_depth": 27, "rf...</td>
          <td>0.483401</td>
          <td>0.436227</td>
        </tr>
        <tr>
          <th>2</th>
          <td>9da5fa7a-610f-469a-9562-b760c03581bc</td>
          <td>{"rf_estimators": 60, "rf_max_depth": 28, "rf_...</td>
          <td>0.480939</td>
          <td>0.450400</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2b63bedb-7983-49cd-8d9b-b2039439ae98</td>
          <td>{"rf_estimators": 233, "rf_max_depth": 28, "rf...</td>
          <td>0.480583</td>
          <td>0.399987</td>
        </tr>
      </tbody>
    </table>
    </div>



Examples of Other Parameter Sets
--------------------------------

Below are some parameters that can be used for **neural networks**,
**`XGBoost <https://en.wikipedia.org/wiki/XGBoost>`__** models,
**fingerprint splits** and
**`ECFP <https://pubs.acs.org/doi/10.1021/ci100050t>`__** features. Each
set of parameters can be used to replace the parameters above. Trying
them out is left as an exercise for the reader.

Neural Network Hyperopt Search
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+------+------+
| Para | Desc |
| mete | ript |
| r    | ion  |
+======+======+
| ``lr | This |
| ``   | cont |
|      | rols |
|      | the  |
|      | lear |
|      | ning |
|      | rate |
|      | .    |
|      | logu |
|      | nifo |
|      | rm\| |
|      | -13. |
|      | 8,-3 |
|      | mean |
|      | s    |
|      | the  |
|      | loga |
|      | rith |
|      | m    |
|      | of   |
|      | the  |
|      | lear |
|      | ning |
|      | rate |
|      | is   |
|      | unif |
|      | orml |
|      | y    |
|      | dist |
|      | ribu |
|      | ted  |
|      | betw |
|      | een  |
|      | -13. |
|      | 8    |
|      | and  |
|      | -3.  |
+------+------+
| ``ls | This |
| ``   | cont |
|      | rols |
|      | laye |
|      | r    |
|      | size |
|      | s.   |
|      | 3\|8 |
|      | ,512 |
|      | mean |
|      | s    |
|      | 3    |
|      | laye |
|      | rs   |
|      | with |
|      | size |
|      | s    |
|      | rang |
|      | ing  |
|      | betw |
|      | een  |
|      | 8    |
|      | and  |
|      | 512  |
|      | neur |
|      | ons. |
|      | A    |
|      | good |
|      | stra |
|      | tegy |
|      | is   |
|      | to   |
|      | star |
|      | t    |
|      | with |
|      | a    |
|      | fewe |
|      | r    |
|      | laye |
|      | rs   |
|      | and  |
|      | slow |
|      | ly   |
|      | incr |
|      | ease |
|      | the  |
|      | numb |
|      | er   |
|      | unti |
|      | l    |
|      | perf |
|      | orma |
|      | nce  |
|      | plat |
|      | eaus |
|      | .    |
+------+------+
| ``dp | This |
| ``   | cont |
|      | rols |
|      | drop |
|      | out. |
|      | 3\|0 |
|      | ,0.4 |
|      | mean |
|      | s    |
|      | 3    |
|      | drop |
|      | out  |
|      | laye |
|      | rs   |
|      | with |
|      | prob |
|      | abil |
|      | ity  |
|      | of   |
|      | zero |
|      | ing  |
|      | a    |
|      | weig |
|      | ht   |
|      | betw |
|      | een  |
|      | 0    |
|      | and  |
|      | 40%. |
|      | This |
|      | need |
|      | s    |
|      | to   |
|      | matc |
|      | h    |
|      | the  |
|      | numb |
|      | er   |
|      | of   |
|      | laye |
|      | rs   |
|      | spec |
|      | ifie |
|      | d    |
|      | with |
|      | ``ls |
|      | ``   |
|      | and  |
|      | shou |
|      | ld   |
|      | rang |
|      | e    |
|      | betw |
|      | een  |
|      | 0%   |
|      | and  |
|      | 50%. |
+------+------+
| ``ma | This |
| x_ep | cont |
| ochs | rols |
| ``   | how  |
|      | long |
|      | to   |
|      | trai |
|      | n    |
|      | each |
|      | mode |
|      | l.   |
|      | Trai |
|      | ning |
|      | for  |
|      | more |
|      | epoc |
|      | hs   |
|      | incr |
|      | ease |
|      | s    |
|      | runt |
|      | ime, |
|      | but  |
|      | allo |
|      | ws   |
|      | mode |
|      | ls   |
|      | more |
|      | time |
|      | to   |
|      | opti |
|      | mize |
|      | .    |
+------+------+

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
^^^^^^^

-  ``xgbg`` Stands for ``xgb_gamma`` and controls the minimum loss
   reduction required to make a further partition on a leaf node of the
   tree.
-  ``xgbl`` Stands for ``xgb_learning_rate`` and controls the boosting
   learning rate searching domain of
   **`XGBoost <https://en.wikipedia.org/wiki/XGBoost>`__** models.

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
^^^^^^^^^^^^^^^^^

This trains an **`XGBoost <https://en.wikipedia.org/wiki/XGBoost>`__**
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
^^^^^^^^^^^^^

This uses an **`XGBoost <https://en.wikipedia.org/wiki/XGBoost>`__**
model with **`ECFP
fingerprints <https://pubs.acs.org/doi/10.1021/ci100050t>`__** features
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
**`AMPL Tutorial Evaluation <https://forms.gle/pa9sHj4MHbS5zG7A6>`__**.
