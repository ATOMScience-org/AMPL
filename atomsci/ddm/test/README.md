# AMPL Tests README

Currently AMPL has 170+ unit tests. About 13% of them use certain resources that will fail if the run environments don't have them. Since 1.4.2 release, AMPL include the checks to bypass the tests if the resources not found. 

---
**NOTE:**
All tests that aren’t listed in the table should run on any machine.
---

**The tests that require special resources**

|| Tests | Notes |
|--| ------ | ----------- |
|1|integrative/compare_models/test_filesystem_perf_results.py::test_AttentiveFP_results()|dgl_required|
|2|integrative/compare_models/test_filesystem_perf_results.py::test_GCN_results()|dgl_required|
|3|integrative/compare_models/test_filesystem_perf_results.py::test_MPNN_results()|dgl_required|
|4|integrative/compare_models/test_filesystem_perf_results.py::test_PytorchMPNN_results()|dgl_required|
|5|integrative/dc_models/test_retrain_dc_models.py::run_test_reg_config_H1_fit_AttentiveFPModel()|dgl_required|
|6|integrative/dc_models/test_retrain_dc_models.py::run_test_reg_config_H1_fit_GCNModel()|dgl_required|
|7|integrative/dc_models/test_retrain_dc_models.py::run_test_reg_config_H1_fit_MPNNModel()|dgl_required|
|8|integrative/dc_models/test_retrain_dc_models.py::run_test_reg_config_H1_fit_PytorchMPNNModel()|dgl_required|
|9|integrative/dc_models/test_dc_models.py::test_reg_config_H1_fit_AttentiveFPModel()|dgl_required|
|10|integrative/dc_models/test_dc_models.py::test_reg_config_H1_fit_GCNModel()|dgl_required|
|11|integrative/dc_models/test_dc_models.py::test_reg_config_H1_fit_MPNNModel()|dgl_required|
|12|integrative/dc_models/test_dc_models.py::test_reg_config_H1_fit_PytorchMPNNModel()|dgl_required|
|13|integrative/delaney_Panel/test_delaney_panel.py::test_reg_config_H1_fit_XGB_moe()|moe_required|
|14|integrative/delaney_Panel/test_delaney_panel.py::test_reg_config_H1_fit_NN_moe()|moe_required|
|15|integrative/delaney_Panel/test_delaney_panel.py::test_reg_config_H1_double_fit_NN_moe()|moe_required|
|16|integrative/delaney_Panel/test_delaney_panel.py::test_multi_class_random_config_H1_fit_NN_moe()|moe_required|
|17|integrative/delaney_Panel/test_delaney_pane::test_class_config_H1_fit_NN_moe()|moe_required
|18|integrative/early_stopping_tests/test_kfold_split.py::test_attentivefp()|dgl_required|
|19|integrative/early_stopping_tests/test_kfold_split.py::test_gcnmodel()|dgl_required|
|20|integrative/early_stopping_tests/test_kfold_split.py::test_mpnnmodel()|dgl_required|
|21|integrative/early_stopping_tests/test_kfold_split.py::test_pytorchmpnnmodel()|dgl_required|
|22|integrative/hybrid/test_hybrid.py::test()|moe_required|
|23|integrative/hyperparam_search/test_hyperparam.py::test()|slurm_required|
|24|integrative/maestro/test_maestro.py::test()|slurm_required|
|25|integrative/shortlist_test/test_shortlist.py::test()|slurm_required|
|26|integrative/seed_test/test_seed_models.py::test_attentivefp_regression_reproducibility()|dgl_required|
|27|integrative/seed_test/test_seed_models.py::test_pytorchmpnn_regression_reproducibility()|dgl_required|
|28|unit/test_LCTimerIterator.py::test_LCTimerKFoldIterator_too_long()|slurm_required|
|29|unit/test_LCTimerIterator.py::test_LCTimerIterator_too_long()|slurm_required|
