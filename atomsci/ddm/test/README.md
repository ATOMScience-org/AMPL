# AMPL Tests README

A few of the AMPL tests use certain resources that will fail if the run environments don't have them.

Since 1.4.2 release, AMPL tests provide the checks to bypass the tests if the resources not found. 

The table below shows the tests and the resources they expect. If you are not running on the Livermore Computing (LC) and some tests fail, it may be one of the sources missing.
___

|| Tests | Notes |
|--| ------ | ----------- |
|1|test_filesystem_perf_results.py::test_MPNN_results() | GPU Required |
|2|test_retrain_dc_models.py::test_reg_config_H1_fit_AttentiveFPModel()|GPU Required|
|3|test_retrain_dc_models.py::test_reg_config_H1_fit_GCNModel()|GPU Required| 
|4|test_retrain_dc_models.py::test_reg_config_H1_fit_MPNNModel()|GPU Required|
|5|test_retrain_dc_models.py::test_reg_config_H1_fit_GraphConvModel()|GPU Required|
|6|test_retrain_dc_models.py::test_reg_config_H1_fit_PytorchMPNNModel()|GPU Required|
|7|test_retrain_dc_models.py::test_reg_config_H1_fit_AttentiveFPModel()|GPU Required|
|8|test_dc_models.py::test_reg_config_H1_fit_AttentiveFPModel()|GPU Required|        
|9|test_delaney_panel.py::test_reg_config_H1_fit_XGB_moe()|GPU, MOE Required|          
|10|test_delaney_panel.py::test_reg_config_H1_fit_NN_moe()|GPU, MOE Required|   
|11|test_delaney_panel.py::test_reg_config_H1_double_fit_NN_moe()|GPU, MOE Required|   
|12|test_delaney_panel.py::test_multi_class_random_config_H1_fit_NN_moe()|GPU, MOE Required|   
|13|test_delaney_panel.py::test_class_config_H1_fit_NN_moe()|GPU, MOE Required|   
|14|test_kfold_split.py|GPU Required| 
|15|test_hybrid.py::test()|MOE required| 
|16|test_hyperparam.py::test()|Slurm used|         
|17|test_maestro.py::test()|Slurm used|   
|18|test_shortlist.py::test()|Slurm used|          
|19|test_wenzel_NN.py::test()|GPU required. mp.create_prediction_pipeline_from_file()| 
|20|test_num_trainable_params.py::test()|GPU required. cm.num_trainable_parameters_from_file()|         
|21|test_prediction_order.py::test_predict_from_model()|GPU Required| 
|22|test_prediction_order.py::test_predict_on_dataframe()|GPU required. mp.create_prediction_pipeline_from_file()|         
|23|test_LCTimerIterator.py::test_LCTimerIterator_too_long()|Slurm used|    
|23|test_LCTimerIterator.py::test_LCTimerKFoldIterator_too_long()|Slurm used|                              