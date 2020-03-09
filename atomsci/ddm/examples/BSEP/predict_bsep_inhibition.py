#!/usr/bin/env python
# coding: utf-8

# Script for running example BSEP inhibition classification models against user-provided data.

import sys
import os
import pandas as pd
import numpy as np
from sklearn import metrics
import argparse

from atomsci.ddm.pipeline import model_pipeline as mp
from atomsci.ddm.pipeline import parameter_parser as parse
from atomsci.ddm.pipeline.perf_data import negative_predictive_value
from atomsci.ddm.utils.struct_utils import base_smiles_from_smiles

# =====================================================================================================
def predict_activity(args):

    input_df = pd.read_csv(args.input_file, index_col=False)
    colnames = set(input_df.columns.values)
    if args.id_col not in colnames:
        input_df['compound_id'] = ['compound_%.6d' % i for i in range(input_df.shape[0])]
        args.id_col = 'compound_id'
    if args.smiles_col not in colnames:
        raise ValueError('smiles_col parameter not specified or column not in input file.')
    if args.dont_standardize:
        std_smiles_col = args.smiles_col
    else:
        print("Standardizing SMILES strings for %d compounds." % input_df.shape[0])
        orig_ncmpds = input_df.shape[0]
        std_smiles = [base_smiles_from_smiles(s) for s in input_df[args.smiles_col].values]
        input_df['standardized_smiles'] = std_smiles
        input_df = input_df[input_df.standardized_smiles != '']
        if input_df.shape[0] == 0:
            print("No valid SMILES strings to predict on.")
            return
        nlost = orig_ncmpds - input_df.shape[0]
        input_df = input_df.sort_values(by=args.id_col)
        orig_smiles = input_df[args.smiles_col].values
        if nlost > 0:
            print("Could not parse %d SMILES strings; will predict on the remainder." % nlost)
        std_smiles_col = 'standardized_smiles'

    pred_params = {
        'id_col': args.id_col,
        'smiles_col': std_smiles_col
    }
    has_activity = (args.activity_col is not None)
    if has_activity:
        pred_params['response_cols'] = args.activity_col
    pred_params = parse.wrapper(pred_params)

    model_files = dict(random = 'bsep_classif_random_split.tar.gz', scaffold = 'bsep_classif_scaffold_split.tar.gz')
    if args.model_type not in model_files:
        raise ValueError("model_type %s is not a recognizied value." % args.model_type)
    
    # Test loading model from tarball and running predictions
    models_dir = os.path.join(os.path.dirname(os.path.dirname(mp.__file__)), 'examples', 'BSEP', 'models')
    model_tarfile = os.path.join(models_dir, model_files[args.model_type])
    pipe = mp.create_prediction_pipeline_from_file(pred_params, reload_dir=None, model_path=model_tarfile)
    pred_df = pipe.predict_full_dataset(input_df, contains_responses=has_activity, dset_params=pred_params)
    pred_df = pred_df.sort_values(by=args.id_col)
    if not args.dont_standardize:
        pred_df[args.smiles_col] = orig_smiles

    # Write predictions to output file
    pred_df.to_csv(args.output_file, index=False)
    print("Wrote predictions to file %s" % args.output_file)

    # If measured activity values are provided, print some performance metrics
    if has_activity:
        actual_vals = pred_df['%s_actual' % args.activity_col].values
        pred_classes = pred_df['%s_pred' % args.activity_col].values
        pred_probs = pred_df['%s_prob' % args.activity_col].values
        conf_matrix = metrics.confusion_matrix(actual_vals, pred_classes)
        roc_auc = metrics.roc_auc_score(actual_vals, pred_probs)
        prc_auc = metrics.average_precision_score(actual_vals, pred_probs)
        accuracy = metrics.accuracy_score(actual_vals, pred_classes)
        precision = metrics.precision_score(actual_vals, pred_classes)
        npv = negative_predictive_value(actual_vals, pred_classes)
        recall = metrics.recall_score(actual_vals, pred_classes)
        mcc = metrics.matthews_corrcoef(actual_vals, pred_classes)
        ncorrect = sum(actual_vals == pred_classes)
        print("Performance metrics:\n")
        print("%d out of %d predictions correct." % (ncorrect, pred_df.shape[0]))
        print("Accuracy: %.3f" % accuracy)
        print("Precision: %.3f" % precision)
        print("Recall: %.3f" % recall)
        print("NPV: %.3f" % npv)
        print("ROC AUC: %.3f" % roc_auc)
        print("PRC AUC: %.3f" % prc_auc)
        print("Matthews correlation coefficient: %.3f" % mcc)
        print("Confusion matrix:")
        print("\t\tpredicted activity")
        print("actual\nactivity\t0\t1\n")
        print("   0\t\t%d\t%d" % (conf_matrix[0][0], conf_matrix[0][1]))
        print("   1\t\t%d\t%d" % (conf_matrix[1][0], conf_matrix[1][1]))



    
# =====================================================================================================
def parse_params():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Script to predict BSEP inhibition for a list of compounds using a classification model.')
    parser.add_argument(
        '--input_file', '-i', dest='input_file', required=True, 
        help='Input CSV file containing list of SMILES strings of compounds to run predictions on.')
    parser.add_argument(
        '--output_file', '-o', dest='output_file', required=True, 
        help='File to write predictions to.')
    parser.add_argument(
        '--model_type', '-m', type=str, dest='model_type', choices=['random', 'scaffold'], default='scaffold',
        help='Label for model to use. Options are "random" or "scaffold" for models trained with random or scaffold split.')
    parser.add_argument(
        '--smiles_col', type=str, dest='smiles_col', default='smiles',
        help='Name of column of input file containing SMILES strings.')
    parser.add_argument(
        '--id_col', type=str, dest='id_col', default='compound_id',
        help='Column of input file containing compound IDs. If none is provided, sequential IDs will be generated.')
    parser.add_argument(
        '--activity_col', type=str, dest='activity_col', default=None,
        help='Optional column of input file containing measured inhibitor activities as binary values, with 1 indicating the compound\n' +
             'is a BSEP inhibitor. If provided, the program will compare the predicted and measured activities and display some performance metrics.')
    parser.add_argument(
        '--dont_standardize', '-s', dest='dont_standardize', action='store_true',
        help="Don't standardize input SMILES strings. By default, the program standardizes SMILES strings and removes any salt groups.")

    parsed_args = parser.parse_args()
    return parsed_args


# =====================================================================================================
def main():
    """Entry point when script is run from a shell"""
    args = parse_params()
    predict_activity(args)


# =====================================================================================================
if __name__ == '__main__' and len(sys.argv) > 1:
    main()
    sys.exit(0)

