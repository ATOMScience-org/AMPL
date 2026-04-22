import pickle
import glob
import os
import pandas as pd
import atomsci.ddm.pipeline.featurization as feat
import atomsci.ddm.pipeline.parameter_parser as pp
import atomsci.ddm.pipeline.model_datasets as model_datasets
import atomsci.ddm.pipeline.transformations as trans
import atomsci.ddm.utils.struct_utils as struct_utils
from deepchem.data import NumpyDataset
import numpy as np
import shutil
import sklearn.utils as sku
import logging
logging.basicConfig(format='%(asctime)-15s %(message)s')
log = logging.getLogger('ATOM')


def prepare_csv_and_descriptor_with_dummy_response(csv_path, descriptor_type, temp_root, split_uuid='split_uuid'):
    """
    Copies the csv file and its descriptor file to a temp directory, preserving structure,
    and adds a 'dummy_response' column of zeros to both.

    Args:
        csv_path (str): Path to the original CSV file.
        descriptor_type (str): Descriptor type to look for in the descriptor file name.
        temp_root (str): Root of the temporary directory to copy files into.
        split_uuid (str): Unique identifier for the split file to look for and copy if it exists.

    Returns:
        (str, str): Paths to the new CSV and descriptor files in the temp directory.
    """
    # Find descriptor file
    csv_dir = os.path.dirname(csv_path)
    csv_base = os.path.splitext(os.path.basename(csv_path))[0]
    descriptor_dir = os.path.join(csv_dir, 'scaled_descriptors')
    descriptor_pattern = f"{csv_base}_with_{descriptor_type}_descriptors.csv"
    descriptor_path = os.path.join(descriptor_dir, descriptor_pattern)
    copy_descriptor_csv =  os.path.exists(descriptor_path)

    # Find split file if it exists
    split_pattern = os.path.join(csv_dir, f'{csv_base}_*_{split_uuid}.csv')
    split_files = glob.glob(split_pattern)
    if len(split_files)>1:
        raise RuntimeError(f'Multiple splits found {split_files}')
    if len(split_files)>0 and len(split_uuid)>0:
        split_csv = split_files[0]
        split_base = os.path.basename(split_csv)
        temp_split_csv = os.path.join(temp_root, split_base)
        shutil.copy(split_csv, temp_split_csv)

    # Prepare destination paths
    temp_csv_path = os.path.join(temp_root, os.path.basename(csv_path))
    temp_descriptor_dir = os.path.join(temp_root, 'scaled_descriptors')
    os.makedirs(temp_descriptor_dir, exist_ok=True)
    temp_descriptor_path = os.path.join(temp_descriptor_dir, os.path.basename(descriptor_path))

    # Copy and add dummy_response to CSV
    df_csv = pd.read_csv(csv_path)
    df_csv['dummy_response'] = 0
    df_csv.to_csv(temp_csv_path, index=False)

    # Copy and add dummy_response to descriptor file
    if copy_descriptor_csv:
        df_desc = pd.read_csv(descriptor_path)
        df_desc['dummy_response'] = 0
        df_desc.to_csv(temp_descriptor_path, index=False)

    return temp_csv_path

def load_all_datasets(
    transformer_dataset_key_configs,
    featurizer,
    descriptor_type
):
    """Loads datasets from configs and builds NumpyDataset

    Args:
        csvs_or_tuples (list): List of csv file paths or (csv_file, split_uuid) tuples.
        featurizer (str): The featurizer type (e.g., 'ecfp', 'graphconv', 'computed_descriptors', etc.).
        descriptor_type (str): Descriptor type (e.g., 'moe', 'rdkit_raw', etc.).

    Returns:
        NumpyDataset
    """
    featurized_datasets = []
    for ds_config in transformer_dataset_key_configs:
        # Prepare params for this dataset
        params_dict = dict()
        params_dict.update(ds_config)
        params_dict['featurizer'] = featurizer
        params_dict['descriptor_type'] = descriptor_type
        params_dict['feature_transform_type'] = 'Identity'
        
        # check if there is a split_uuid in the config
        split_uuid = params_dict.get('split_uuid', None)

        params = pp.wrapper(params_dict)
        dataset = model_datasets.create_and_load_model_dataset(params, ds_client=None)

        # If split_uuid is provided, use only the training subset
        if split_uuid:
            dataset.split_dataset()
            train_dset = dataset.train_valid_dsets[0][0]
            featurized_datasets.append(train_dset)
        else:
            # this is a NumpyDataset with all data
            featurized_datasets.append(dataset.dataset)
        
    # Combine all dataframes for fitting transformers
    combined_dataset = NumpyDataset(
        X=np.vstack([d.X for d in featurized_datasets]),
        y=np.vstack([d.y for d in featurized_datasets]),
        ids=np.concatenate([d.ids for d in featurized_datasets]),
        w=np.concatenate([d.w for d in featurized_datasets])
    )

    return combined_dataset

def filter_outlier_features(dataset_key, id_col, smiles_col, response_cols, featurizer, descriptor_type, threshold=1e10):
    """Looks for compounds with very large descriptor values.

    Args:
        dataset_key_configs (list): List of dataset key configuration dictionaries.
        featurizer (str): The featurizer type (e.g., 'ecfp', 'graphconv', 'computed_descriptors', etc.).
        descriptor_type (str): Descriptor type (e.g., 'moe', 'rdkit_raw', etc.).
        threshold (float): Threshold for filtering large descriptor values.

    Returns:
        DataFrame with outlier compound_ids, descriptor column names, and descriptor values that exceed the threshold.
    """
    dataset_key_config = {
        'dataset_key': dataset_key,
        'id_col': id_col,
        'smiles_col': smiles_col,
        'response_cols': response_cols
    }
    params_dict = dict()
    params_dict.update(dataset_key_config)
    params_dict['featurizer'] = featurizer
    params_dict['descriptor_type'] = descriptor_type
    params_dict['feature_transform_type'] = 'Identity'
    params_dict['verbose'] = True
    
    params = pp.wrapper(params_dict)
    dataset = model_datasets.create_and_load_model_dataset(params, ds_client=None)

    abs_X = np.abs(dataset.dataset.X)
    try:
        sku.assert_all_finite(abs_X)
    except ValueError:
        # data contains inf or nan values
        log.warning("SklearnPipelineWrapper: data contains NaN or Inf; replacing with zeros")
        abs_X = trans.zero_out_inf_nan(abs_X)
   

    large_values = np.argwhere(abs_X > threshold)
    feature_cols = np.array(dataset.featurization.get_feature_columns())

    return dataset.dataset.ids[large_values[:,0]], [feature_cols[i] for i in large_values[:,1]], [abs_X[i,j] for i,j in large_values]

def filter_outlier_MW(dataset_key, smiles_col, threshold=1000, workers=8):
    """Filters datasets and looks for compounds with very large molecular weights.

    Args:
        dataset_key (str): Path to the dataset CSV file.
        smiles_col (str): Name of the column containing SMILES strings.
        threshold (float): Threshold for filtering large molecular weights.
        workers (int): Number of workers to use for parallel processing in calculating molecular weights.
        workers (int): Number of workers to use for parallel processing in calculating molecular weights.

    Returns:
        List of SMILES with molecular weights that exceed the threshold.

    """
    # molecular weight > 1000
    # calculate molecular weight here.
    dataset_df = pd.read_csv(dataset_key)
    mol_weights = struct_utils.mol_wt_from_smiles(dataset_df[smiles_col].to_list(), workers=workers)
    mw_large = np.argwhere(np.array(mol_weights) > threshold)

    return dataset_df[smiles_col].iloc[mw_large.flatten()].tolist()

def build_and_save_feature_transformers_from_csvs(
    transformer_dataset_key_configs,
    dest_pkl_path,
    featurizer,
    descriptor_type,
    feature_transform_type,
    **kwargs
):
    """
    Build feature transformers_x from a list of csv files (or csv+split_uuid tuples) and save them as a pickle file,
    including the params object used to create them. This function saves feature transformers suitable for use
    for one fold. 

    Args:
        transformer_dataset_key_configs (list): A list of dictionaries that contain information about each dataset_key such
        as id_col, smiles_col, response_cols, split_uuids, etc.
        dest_pkl_path (str): Path to save the pickle file.
        featurizer (str): The featurizer type (e.g., 'ecfp', 'graphconv', 'computed_descriptors', etc.).
        descriptor_type (str): Descriptor type (e.g., 'moe', 'rdkit_raw', etc.).
        feature_transform_type (str): The type of transformer to use (e.g., 'RobustScaler', 'PowerTransformer', etc.).
        **kwargs: Additional keyword arguments for transformer params.

    Returns:
        None
    """
    combined_dataset = load_all_datasets(transformer_dataset_key_configs=transformer_dataset_key_configs, featurizer=featurizer, descriptor_type=descriptor_type)

    # Build a single params object for transformer fitting
    params_dict = dict(
        featurizer=featurizer,
        feature_transform_type=feature_transform_type,
        descriptor_type=descriptor_type,
        **kwargs
    )
    params = pp.wrapper(params_dict)
    featurization = feat.create_featurization(params)

    # Build the feature transformers (transformers_x)
    transformers_x = featurization.create_feature_transformer(combined_dataset, params)

    # Save both transformers_x and params to a pickle file
    # copy processed params back into params_dict in case wrapper
    # updates them.
    for k in params_dict.keys():
        if k in params.__dict__:
            params_dict[k] = params.__dict__[k]

    params_dict['transformer_dataset_key_configs'] = transformer_dataset_key_configs
    with open(dest_pkl_path, 'wb') as f:
        pickle.dump({'transformers_x': transformers_x, 'params': params_dict}, f)

    print(f"Feature transformers_x and params saved to {dest_pkl_path}")