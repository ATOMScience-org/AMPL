#!/usr/bin/env python

# Script for running example BSEP inhibition classification models against user-provided data.

import argparse
import os
import sys

import pandas as pd
from sklearn import metrics

from atomsci.ddm.pipeline import predict_from_model as pfm

# from atomsci.ddm.pipeline import model_pipeline as mp
# from atomsci.ddm.pipeline import parameter_parser as parse
from atomsci.ddm.pipeline.perf_data import negative_predictive_value

# from atomsci.ddm.utils.struct_utils import base_smiles_from_smiles

# =====================================================================================================
def predict_activity(args):
    # prepare inputs
    input_df = pd.read_csv(args.input_file, index_col=False)
    colnames = set(input_df.columns.values)
    if args.id_col not in colnames:
        input_df['compound_id'] = [f'compound_{i:06d}' for i in range(input_df.shape[0])]
        args.id_col = 'compound_id'
    if args.smiles_col not in colnames:
        raise ValueError('smiles_col parameter not specified or column not in input file.')
    model_files = {'random': 'bsep_classif_random_split.tar.gz', 'scaffold': 'bsep_classif_scaffold_split.tar.gz'}
    if args.model_type not in model_files:
        raise ValueError(f"model_type {args.model_type} is not a recognizied value.")
    if args.external_training_data is not None:
        data_file = os.path.join(os.getcwd(), args.external_training_data)
    else:
        data_file=None
    print("Data file:", data_file)
    # Test loading model from tarball and running predictions
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    model_tarfile = os.path.join(models_dir, model_files[args.model_type])
    
    # predict
    pred_df = pfm.predict_from_model_file(model_path = model_tarfile, input_df=input_df, id_col=args.id_col, smiles_col=args.smiles_col, response_col=args.response_col, dont_standardize=args.dont_standardize, is_featurized = args.is_featurized, AD_method=args.AD_method, external_training_data=data_file)
    
    # delete files created during prediction
    for root, dirs, files in os.walk(os.getcwd(), topdown=False):
        for file in files:
            if 'train_valid_test' in file:
                os.remove(os.path.join(root,file)) 
    
    # Write predictions to output file
    pred_df.to_csv(args.output_file, index=False)
    print(f"Wrote predictions to file {args.output_file}")

    # If measured activity values are provided, print some performance metrics
    if args.response_col is not None:
        actual_vals = pred_df[f'{args.response_col}_actual'].values
        pred_classes = pred_df[f'{args.response_col}_pred'].values
        pred_probs = pred_df[f'{args.response_col}_prob'].values
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
        print(f"{ncorrect} out of {pred_df.shape[0]} predictions correct.")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"NPV: {npv:.3f}")
        print(f"ROC AUC: {roc_auc:.3f}")
        print(f"PRC AUC: {prc_auc:.3f}")
        print(f"Matthews correlation coefficient: {mcc:.3f}")
        print("Confusion matrix:")
        print("\t\tpredicted activity")
        print("actual\nactivity\t0\t1\n")
        print(f"   0\t\t{conf_matrix[0][0]}\t{conf_matrix[0][1]}")
        print(f"   1\t\t{conf_matrix[1][0]}\t{conf_matrix[1][1]}")



    
# =====================================================================================================
def parse_params():
    """Parse command line arguments"""
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
        '--activity_col', type=str, dest='response_col', default=None,
        help='Optional column of input file containing measured inhibitor activities as binary values, with 1 indicating the compound\n' +
             'is a BSEP inhibitor. If provided, the program will compare the predicted and measured activities and display some performance metrics.')
    parser.add_argument(
        '--ad_method', type=str, dest='AD_method', default='z_score',
        help='Method to calculate accessibility domain index. z_score or local_density are accepted. Will not work without also passing original model training data file path as ext_train_data')
    parser.add_argument(
        '--ext_train_data', type=str, dest='external_training_data', default=None,
        help='This path should point to the original dataset the model was trained on, which will be used to calculate the AD index.')
    parser.add_argument(
        '--is_featurized', '-f', dest='is_featurized', action='store_true',
        help='Data is already featurized. You have already calculated the correct features for this model and are passing a featurized df. By default, the program will featurize the data for you - may be slow depending on the number of compounds.')
    parser.add_argument(
        '--dont_standardize', '-s', dest='dont_standardize', action='store_true',
        help="Don't standardize input SMILES strings. By default, the program standardizes SMILES strings and removes any salt groups.")
    parser.add_argument(
        '--result_dir', type=str, dest='result_dir', default='./bsep_result',
        help='directory to save the results.')

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

