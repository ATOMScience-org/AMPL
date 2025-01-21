import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from atomsci.ddm.pipeline.diversity_plots import diversity_plots
from rdkit import Chem

@pytest.fixture
def sample_data():
    data = {
        'compound_id': ['cmpd1', 'cmpd2', 'cmpd3'],
        'rdkit_smiles': ['CCO', 'CCN', 'CCC'],
        'response': [1, 0, 1]
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_datastore_functions(monkeypatch):
    def mock_retrieve_dataset_by_datasetkey(dset_key, bucket):
        return pd.DataFrame({
            'compound_id': ['cmpd1', 'cmpd2', 'cmpd3'],
            'rdkit_smiles': ['CCO', 'CCN', 'CCC'],
            'response': [1, 0, 1]
        })
    monkeypatch.setattr('atomsci.ddm.utils.datastore_functions.retrieve_dataset_by_datasetkey', mock_retrieve_dataset_by_datasetkey)

@pytest.fixture
def mock_struct_utils(monkeypatch):
    def mock_base_mol_from_smiles(smiles):
        return Chem.MolFromSmiles(smiles)
    monkeypatch.setattr('atomsci.ddm.utils.struct_utils.base_mol_from_smiles', mock_base_mol_from_smiles)

@pytest.fixture
def mock_dist_metrics(monkeypatch):
    def mock_mcs(mols):
        return np.random.rand(len(mols), len(mols))
    def mock_tanimoto(fps):
        return np.random.rand(len(fps), len(fps))
    monkeypatch.setattr('atomsci.ddm.pipeline.dist_metrics.mcs', mock_mcs)
    monkeypatch.setattr('atomsci.ddm.pipeline.dist_metrics.tanimoto', mock_tanimoto)

@pytest.fixture
def mock_plotting(monkeypatch):
    monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
    monkeypatch.setattr('matplotlib.backends.backend_pdf.PdfPages', lambda x: None)

def test_diversity_plots(sample_data, mock_datastore_functions, mock_struct_utils, mock_dist_metrics, mock_plotting):
    with patch('atomsci.ddm.utils.datastore_functions.retrieve_dataset_by_datasetkey') as mock_retrieve:
        mock_retrieve.return_value = sample_data
        diversity_plots(
            dset_key='../test_datasets/H1_std.csv',
            datastore=False,
            bucket='public',
            title_prefix='Test',
            ecfp_radius=2,
            umap_file=None,
            out_dir=None,
            id_col='compound_id',
            smiles_col='base_rdkit_smiles',
            is_base_smiles=False,
            max_for_mcs=300,
            colorpal=None
        )
        mock_retrieve.assert_called_once_with('../test_datasets/H1_std.csv', 'public')