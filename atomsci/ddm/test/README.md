# AMPL Tests README

Currently AMPL has 170+ unit tests. About 13% of them use certain resources that will fail if the run environments don't have them. Since 1.4.2 release, AMPL include the checks to bypass the tests if the resources not found. 

---
**NOTE:**
All tests that aren’t listed in the table should run on any machine.
---

**The tests that require special resources**

|| Tests | Notes |
|--| ------ | ----------- |
|1|test_filesystem_perf_results.py::test_MPNN_results() | DGL Required |
|2|test_retrain_dc_models.py::test_reg_config_H1_fit_AttentiveFPModel()|DGL Required|
|3|test_retrain_dc_models.py::test_reg_config_H1_fit_GCNModel()|DGL Required|
|4|test_retrain_dc_models.py::test_reg_config_H1_fit_MPNNModel()|DGL Required|
|5|test_retrain_dc_models.py::test_reg_config_H1_fit_PytorchMPNNModel()|DGL Required|
|6|test_retrain_dc_models.py::test_reg_config_H1_fit_AttentiveFPModel()|DGL Required|
|7|test_dc_models.py::test_reg_config_H1_fit_AttentiveFPModel()|DGL Required|
|8|test_delaney_panel.py::test_reg_config_H1_fit_XGB_moe()|MOE Required|
|9|test_delaney_panel.py::test_reg_config_H1_fit_NN_moe()|MOE Required|
|10|test_delaney_panel.py::test_reg_config_H1_double_fit_NN_moe()|MOE Required|
|11|test_delaney_panel.py::test_multi_class_random_config_H1_fit_NN_moe()|MOE Required|
|12|test_delaney_panel.py::test_class_config_H1_fit_NN_moe()|MOE Required|
|13|test_hybrid.py::test()|MOE required|
|14|test_hyperparam.py::test()|Slurm used|
|15|test_maestro.py::test()|Slurm used|
|16|test_shortlist.py::test()|Slurm used|
|17|test_LCTimerIterator.py::test_LCTimerIterator_too_long()|Slurm used|
|18|test_LCTimerIterator.py::test_LCTimerKFoldIterator_too_long()|Slurm used|