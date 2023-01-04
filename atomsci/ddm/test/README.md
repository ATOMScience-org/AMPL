# AMPL Tests README

Currently AMPL has 170+ unit tests. About 13% of them use certain resources that will fail if the run environments don't have them. Since 1.4.2 release, AMPL include the checks to bypass the tests if the resources not found. 

---
**NOTE:**
All tests that aren’t listed in the table should run on any machine.
---

**The tests that require special resources**

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
|14|test_kfold_split.py|Exceeded free system memory|
|15|test_hybrid.py::test()|MOE required| 
|16|test_hyperparam.py::test()|Slurm used|         
|17|test_maestro.py::test()|Slurm used|   
|18|test_shortlist.py::test()|Slurm used|          
|19|test_wenzel_NN.py::test()|Could not download dataset|
|20|test_num_trainable_params.py::test()|GPU required|
|21|test_prediction_order.py::test_predict_from_model()|GPU Required|
|22|test_prediction_order.py::test_predict_on_dataframe()|GPU required|
|23|test_LCTimerIterator.py::test_LCTimerIterator_too_long()|Slurm used|
|23|test_LCTimerIterator.py::test_LCTimerKFoldIterator_too_long()|Slurm used|