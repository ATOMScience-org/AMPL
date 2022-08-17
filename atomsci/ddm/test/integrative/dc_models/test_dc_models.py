#!/usr/bin/env python

from test_retrain_dc_models import *
from atomsci.ddm.utils import llnl_utils

# Train and Predict
# -----
def test_reg_config_H1_fit_AttentiveFPModel():
    if not llnl_utils.is_lc_system():
        assert True
        return
        
    H1_init()
    train_and_predict('reg_config_H1_fit_AttentiveFPModel.json', prefix='H1') # crashes during run
# -----
def test_reg_config_H1_fit_GCNModel():
    H1_init()
    train_and_predict('reg_config_H1_fit_GCNModel.json', prefix='H1') # crashes during run
# -----
def test_reg_config_H1_fit_MPNNModel():
    H1_init()
    train_and_predict('reg_config_H1_fit_MPNNModel.json', prefix='H1') # crashes during run

def test_reg_config_H1_fit_GraphConvModel():
    H1_init()
    train_and_predict('reg_config_H1_fit_GraphConvModel.json', prefix='H1') # crashes during run

def test_reg_config_H1_fit_PytorchMPNNModel():
    H1_init()
    train_and_predict('reg_config_H1_fit_PytorchMPNNModel.json', prefix='H1') # crashes during run

if __name__ == '__main__':
    test_reg_config_H1_fit_PytorchMPNNModel() # Pytorch implementation of MPNNModel
    test_reg_config_H1_fit_GraphConvModel() # the same model as graphconv
    test_reg_config_H1_fit_MPNNModel() # uses the WeaveFeaturizer
    test_reg_config_H1_fit_GCNModel()
    test_reg_config_H1_fit_AttentiveFPModel() #works fine?