#!/usr/bin/env python
import json
import numpy as np
import pandas as pd
import os
import sys
import shutil
import glob

import atomsci.ddm.pipeline.model_pipeline as mp
import atomsci.ddm.pipeline.predict_from_model as pfm
import atomsci.ddm.pipeline.parameter_parser as parse
import atomsci.ddm.utils.curate_data as curate_data
import atomsci.ddm.utils.struct_utils as struct_utils

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import integrative_utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '../delaney_Panel'))
import test_delaney_panel as tdp

def clean():
    files = glob.glob('delaney-processed*')
    for f in files:
        if os.path.isfile(f):
            os.remove(f)

    if os.path.exists('result'):
        shutil.rmtree('result')
    if os.path.exists('scaled_descriptors'):
        shutil.rmtree('scaled_descriptors')

def test_RF():
    # init training dataset
    init()
    # Train model
    # -----------
    # Read parameter JSON file
    prefix = 'delaney-processed'
    num_iterations = 10
    with open('reg_config_delaney_fit_RF_mordred_filtered.json') as f:
        config = json.loads(f.read())

    r2_scores = []
    for i in range(num_iterations):
        # Parse parameters
        params = parse.wrapper(config)

        # Create model pipeline
        model = mp.ModelPipeline(params)

        # Train model
        model.train_model()

        # Get uuid and reload directory
        # -----------------------------
        model_type = params.model_type
        prediction_type = params.prediction_type
        descriptor_type = params.descriptor_type
        featurizer = params.featurizer
        splitter = params.splitter
        model_dir = 'result/%s_curated_fit/%s_%s_%s_%s'%(prefix, model_type, featurizer, splitter, prediction_type)
        uuid = model.params.model_uuid
        tar_f = 'result/%s_curated_fit_model_%s.tar.gz'%(prefix, uuid)
        reload_dir = model_dir+'/'+uuid

        # check saved scores
        score = integrative_utilities.read_training_statistics_file(reload_dir, 'test', 'r2_score')
        r2_scores.append(score)

    print(r2_scores)
    assert(np.std(r2_scores) > .001), "Not enough variation in random forests."

    # clean again
    clean()

def test_seeded_RF():
    # init training dataset
    init()
    # Train model
    # -----------
    # Read parameter JSON file
    prefix = 'delaney-processed'
    num_iterations = 10
    with open('reg_config_delaney_fit_RF_mordred_filtered.json') as f:
        config = json.loads(f.read())

    config['rf_random_seed'] = 0

    r2_scores = []
    for i in range(num_iterations):
        # Parse parameters
        params = parse.wrapper(config)

        # Create model pipeline
        model = mp.ModelPipeline(params)

        # Train model
        model.train_model()

        # Get uuid and reload directory
        # -----------------------------
        model_type = params.model_type
        prediction_type = params.prediction_type
        descriptor_type = params.descriptor_type
        featurizer = params.featurizer
        splitter = params.splitter
        model_dir = 'result/%s_curated_fit/%s_%s_%s_%s'%(prefix, model_type, featurizer, splitter, prediction_type)
        uuid = model.params.model_uuid
        tar_f = 'result/%s_curated_fit_model_%s.tar.gz'%(prefix, uuid)
        reload_dir = model_dir+'/'+uuid

        # check saved scores
        score = integrative_utilities.read_training_statistics_file(reload_dir, 'test', 'r2_score')
        r2_scores.append(score)

    print(r2_scores)
    assert(np.std(r2_scores) < .001), "Too much variation in random forests."

    # clean again
    clean()

def init():
    """
    Test full model pipeline: Curate data, fit model, and predict property for new compounds
    """

    # Clean
    # -----
    clean()

    # Download
    # --------
    tdp.download()

    # Curate
    # ------
    tdp.curate()

if __name__ == '__main__':
    test_RF()
    test_seeded_RF()
    #pass
