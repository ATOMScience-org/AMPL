import pickle
import glob
import os
import pandas as pd
import atomsci.ddm.pipeline.featurization as feat
import atomsci.ddm.pipeline.parameter_parser as pp
import atomsci.ddm.pipeline.model_datasets as model_datasets
from deepchem.data import NumpyDataset
import numpy as np
import tempfile
import shutil

def prepare_csv_and_descriptor_with_dummy_response(csv_path, descriptor_type, temp_root, split_uuid='split_uuid'):
    """
    Copies the csv file and its descriptor file to a temp directory, preserving structure,
    and adds a 'dummy_response' column of zeros to both.

    Args:
        csv_path (str): Path to the original CSV file.
        descriptor_type (str): Descriptor type to look for in the descriptor file name.
        temp_root (str): Root of the temporary directory to copy files into.

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
    dataset_key_configs,
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
    for ds_config in dataset_key_configs:
        # Prepare params for this dataset
        params_dict = dict()
        params_dict.update(ds_config)
        params_dict['featurizer'] = featurizer
        params_dict['descriptor_type'] = descriptor_type
        params_dict['feature_transform_type'] = 'Identity'
        
        # check if there is a split_uuid in the config
        split_uuid = params_dict.get('split_uuid', None)

        params = pp.wrapper(params_dict)
        featurization = feat.create_featurization(params)
        dataset = model_datasets.create_model_dataset(params, featurization, ds_client=None)

        # Load featurized data (this will use scaled_descriptors if available)
        dataset.get_featurized_data()

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


def build_and_save_feature_transformers_from_csvs(
    dataset_key_configs,
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
        dataset_key_configs (list): A list of dictionaries that contain information about each dataset_key such
        as id_col, smiles_col, response_cols, split_uuids, etc.
        dest_pkl_path (str): Path to save the pickle file.
        featurizer (str): The featurizer type (e.g., 'ecfp', 'graphconv', 'computed_descriptors', etc.).
        descriptor_type (str): Descriptor type (e.g., 'moe', 'rdkit_raw', etc.).
        feature_transform_type (str): The type of transformer to use (e.g., 'RobustScaler', 'PowerTransformer', etc.).
        **kwargs: Additional keyword arguments for transformer params.

    Returns:
        None
    """
    combined_dataset = load_all_datasets(dataset_key_configs=dataset_key_configs, featurizer=featurizer, descriptor_type=descriptor_type)

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

    params_dict['dataset_key_configs'] = dataset_key_configs
    with open(dest_pkl_path, 'wb') as f:
        pickle.dump({'transformers_x': transformers_x, 'params': params_dict}, f)

    print(f"Feature transformers_x and params saved to {dest_pkl_path}")