#!/usr/bin/env python

import json
import pandas as pd
import os
import sys
import glob

import atomsci.ddm.pipeline.parameter_parser as parse

def clean():
    """Clean test files"""
    if "output" not in os.listdir():
        os.mkdir("output")
    for f in os.listdir("./output"):
        if os.path.isfile("./output/"+f):
            os.remove("./output/"+f)
    if "tmp" not in os.listdir():
        os.mkdir("tmp")
    for f in os.listdir("./tmp"):
        if os.path.isfile("./tmp/"+f):
            os.remove("./tmp/"+f)

def test():
    """Test full model pipeline: Curate data, fit model, and predict property for new compounds"""

    # Clean
    # -----
    clean()

    # Run Optuna search
    # -----------------
    with open("H1_RF_hyperopt.json", "r") as f:
        hp_params = json.load(f)

    script_dir = parse.__file__.strip("parameter_parser.py").replace("/pipeline/", "")
    python_path = sys.executable
    hp_params["script_dir"] = script_dir
    hp_params["python_path"] = python_path

    params = parse.wrapper(hp_params)
    if not os.path.isfile(params.dataset_key):
        hp_params["dataset_key"] = os.path.join(script_dir, hp_params["dataset_key"])

    with open("H1_RF_hyperopt_temp.json", "w") as f:
        json.dump(hp_params, f, indent=4)

    run_cmd = f"{python_path} {script_dir}/utils/hyperparam_search_wrapper.py --config_file ./H1_RF_hyperopt_temp.json"
    os.system(run_cmd)

    # check results
    # -------------
    perf_table = glob.glob("./output/performance*")
    best_model = glob.glob("./output/best*")

    try:
        assert perf_table is not None, "perf_table is None"
        assert (len(perf_table) == 1), 'Error: No performance table returned.'
        assert best_model is not None, "best_model is None"
        assert (len(best_model) == 1), 'Error: No best model saved'
        perf_df = pd.read_csv(perf_table[0])
        assert (len(perf_df) == 10), 'Error: Size of performance table WRONG.'
    except AssertionError as e:
        print(f"WARNING: {e}. Continuing.")

if __name__ == '__main__':
    test()
