from argparse import Namespace
import pytest
import numpy as np
import atomsci.ddm.pipeline.featurization as feat
import deepchem as dc
from atomsci.ddm.pipeline import model_datasets as md
import pandas as pd
import unittest.mock as mock
import atomsci.ddm.pipeline.parameter_parser as param_parser
try:
    from mol_vae_features import MoleculeVAEFeaturizer
    mol_vae_supported = True
except ModuleNotFoundError:
    mol_vae_supported = False

import utils_testing as utils

#WARNING: assuming model_dataset.py can create a functional data object.
#WARNING: assuming that config_delaney.json and config_datastore_cav12.json are in the current directory. 

ownership = 'gskusers-ad'
datastore_is_down = utils.datastore_status()

(delaney_params_ecfp, data_obj_ecfp, df_delaney) = utils.delaney_objects()
featurizer_ecfp = data_obj_ecfp.featurization
md.check_task_columns(delaney_params_ecfp, df_delaney)

(delaney_params_graphconv, data_obj_graphconv, df_delaney) = utils.delaney_objects(featurizer="graphconv")
featurizer_graphconv = data_obj_graphconv.featurization
md.check_task_columns(delaney_params_graphconv, df_delaney)

if mol_vae_supported:
    (delaney_params_molvae, data_obj_molvae, df_delaney) = utils.delaney_objects(featurizer="molvae")
    featurizer_molvae = data_obj_molvae.featurization

if not datastore_is_down:
    (datastore_params, mdl_datastore, df_datastore) = utils.datastore_objects()

"""
#TODO: Will implement testing for MOE descriptors when there is more computational bandwidth (takes up to 5-10 minutes to load up the descriptors on a CPU)
#Note: dragon7 is not yet implemented. Only moe descriptors are supported

"""

#***********************************************************************************  
def test_remove_duplicate_smiles(caplog):
    """checking for removal of duplicates and activity with dataframes with no dupes"""
    #checking for a file with no dupes
    no_dupe_df = feat.remove_duplicate_smiles(df_delaney,smiles_col=delaney_params_ecfp.smiles_col)
    test = []
    test.append((len(no_dupe_df) <= len(df_delaney)))
    test.append(len(set(no_dupe_df[delaney_params_ecfp.smiles_col].values.tolist())) <= len(no_dupe_df))
    
    #checking for a file with dupes
    # TODO (ksm): Replace the CaV1.2 dataset I plugged in for datastore testing with a non-ML-ready dataset with dupes.
    if not datastore_is_down:
        no_dupe_df = feat.remove_duplicate_smiles(df_datastore,smiles_col=datastore_params.smiles_col)
        test.append(len(no_dupe_df) <= len(df_datastore))
        test.append(len(set(no_dupe_df[datastore_params.smiles_col].values.tolist())) <= len(no_dupe_df))
    assert all(test)
#***********************************************************************************

def test_create_featurization_dynamicfeaturization():
    """testing if classes are properly generated from the factory method. Asserting that the correct methods exist, and are callable. Asserting correct dc featurizer object is called"""
    test = []
    test.append(isinstance(featurizer_ecfp, feat.DynamicFeaturization))
    methods = ["featurize_data","get_feature_columns","extract_prefeaturized_data","get_feature_count","get_feature_specific_metadata","get_featurized_dset_name","get_featurized_data_subdir"]
    for method in methods:
        test.append(callable(getattr(featurizer_ecfp,method)))
        
    test.append(isinstance(featurizer_ecfp.featurizer_obj, dc.feat.CircularFingerprint))
    
    test.append(isinstance(featurizer_graphconv.featurizer_obj, dc.feat.graph_features.ConvMolFeaturizer))    

    if mol_vae_supported:
        test.append(isinstance(featurizer_molvae.featurizer_obj, MoleculeVAEFeaturizer))
    
    assert all(test)
        

    
#***********************************************************************************    
def test_get_feature_columns_dynamicfeaturization():
    """Testing that dynamic featurization is pulling out the correct number of features and 'column names'. Also technically testing get_feature_count"""
    test = []
    cols = featurizer_ecfp.get_feature_columns()
    test.append(len(cols) == delaney_params_ecfp.ecfp_size)
    test.append(cols[0] == 'c0')
    
    cols = featurizer_graphconv.get_feature_columns()
    test.append(len(cols) == 75)
    
    if mol_vae_supported:
        cols = featurizer_molvae.get_feature_columns()
        test.append(len(cols) == 292)
    assert all(test)

#***********************************************************************************
def test_get_featurized_dset_name_dynamicfeaturization():
    """Dynamic featurization does not support get_featurized_dset_name"""
    with pytest.raises(Exception):
        featurizer_ecfp.get_featurized_dset_name(data_obj_ecfp.dataset_name)
#***********************************************************************************

def test_get_featurized_data_subdir_dynamicfeaturization():
    """Dynamic featurization does not support get_featurized_data_subdir"""
    with pytest.raises(Exception):
        featurizer_ecfp.get_featurized_data_subdir()
#***********************************************************************************

def test_get_feature_specific_metadata_dynamicfeaturization():
    """Dynamic featurization returns a dictionary of parameter settings of the featurization object that are specific to the feature type. Testing all three currently implemented featurizers (ecfp, graphconv, molvae)"""
    ecfp_metadata = featurizer_ecfp.get_feature_specific_metadata(delaney_params_ecfp)
    assert ecfp_metadata =={'ecfp_specific':
                            {"ecfp_radius": delaney_params_ecfp.ecfp_radius,
                             "ecfp_size":delaney_params_ecfp.ecfp_size }}
    graphconv_metadata = featurizer_graphconv.get_feature_specific_metadata(delaney_params_graphconv)
    assert graphconv_metadata == {}
   
#***********************************************************************************
def get_featurized_dset_name_descriptorfeaturization():
    #sanity check that we are getting the correct featurized dset name. Also generating the featurized dset name for saving the featurized dataset. Used in model_datasets.save_featurized_data
    featurized_dset_name = featurizer_desc.get_featurized_dset_name(desc_data_obj.dataset_name)
    assert featurized_dset_name == "subset_" + featurizer_desc.descriptor_key + "_" + featurizer_desc.dataset_name + ".csv"
    
#***********************************************************************************
def get_featurized_data_subdir_descriptorfeaturization():
    #sanity check for gtting the name of the subdirectory. used in model_dataset.save_featurized_data
    assert featurizer_desc.get_featurized_data_subdir() == "scaled_descriptors"
#***********************************************************************************
def get_get_feature_columns_and_count_descriptorfeaturization():
    #sanity check for getting the feature columns. Also testing get_feature_count. Since dragon7 is not yet implemented, will not test for those columns.
    test = []
    moe_desc_cols = featurizer_desc.get_feature_columns() 
    test.append(moe_desc_cols[0] == 'ASA+_per_atom')
    test.append(len(moe_desc_cols) == 306 and featurizer_desc.get_feature_count() == 306)
    moe_desc_cols_all = featurizer_desc.get_feature_columns(include_all=True) 
    excluded_moe_desc_cols = ['E', 'E_ang', 'E_ele', 'E_nb', 'E_oop', 'E_sol', 'E_stb', 'E_str', 'E_strain', 'E_tor', 'E_vdw']
    test.append(set(excluded_moe_desc_cols).intersection(set(moe_desc_cols_all)) == set(excluded_moe_desc_cols))
    test.append(len(excluded_moe_desc_cols) == 317 and featurizer_desc.get_feature_count(include_all=False) == 317)
    assert all(test)
    
#***********************************************************************************

def test_get_mordred_calculator():
    try:
        from mordred import Calculator, descriptors,get_descriptors_from_module # noqa: F401
        from mordred.EState import AtomTypeEState, AggrType
        from mordred.MolecularDistanceEdge import MolecularDistanceEdge
        from mordred import BalabanJ, BertzCT, HydrogenBond, MoeType, RotatableBond, SLogP, TopoPSA # noqa: F401
        #rdkit_desc_mods = [BalabanJ, BertzCT, HydrogenBond, MoeType, RotatableBond, SLogP, TopoPSA]
        mordred_supported = True

        default = feat.get_mordred_calculator()
        assert len(default.descriptors) == 1556

        no_ring_feat = feat.get_mordred_calculator(exclude=feat.subclassed_mordred_classes+['RingCount'])
        assert len(no_ring_feat.descriptors) == 1418

        # no mordred features
        # get_mordred_calculator prepends mordred. for you
        exclude = [d.__name__.replace('mordred.', '') for d in descriptors.all]
        no_mordred_feat = feat.get_mordred_calculator(exclude=exclude)
        assert len(no_mordred_feat.descriptors) == 65

    except ImportError:
        mordred_supported = False   

#***********************************************************************************
def test_copy_featurizer_params():
    """Test the copy_featurizer_params function to ensure it correctly copies featurization parameters."""

    # Create source and destination Namespace objects
    source = Namespace(
        descriptor_type="mordred",
        ecfp_radius=3,
        ecfp_size=2048,
        featurizer="ecfp",
        mordred_cpus=4
    )
    dest = Namespace(
        descriptor_type="rdkit",
        ecfp_radius=2,
        ecfp_size=1024,
        featurizer="graphconv",
        mordred_cpus=2
    )

    # Call the function
    result = feat.copy_featurizer_params(source, dest)

    # Assert that the result is a deepcopy and not the same object as dest
    assert result is not dest

    # Assert that the featurization parameters were correctly copied
    assert result.descriptor_type == source.descriptor_type
    assert result.ecfp_radius == source.ecfp_radius
    assert result.ecfp_size == source.ecfp_size
    assert result.featurizer == source.featurizer
    assert result.mordred_cpus == source.mordred_cpus

    # Assert that other attributes in dest remain unchanged
    assert hasattr(result, "descriptor_type")
    assert hasattr(result, "ecfp_radius")
    assert hasattr(result, "ecfp_size")
    assert hasattr(result, "featurizer")
    assert hasattr(result, "mordred_cpus")

#***********************************************************************************
@pytest.mark.moe_required
def test_compute_moe():
    """Test the compute_all_moe_descriptors function to ensure it correctly generates moe features."""
    smiles_df = pd.DataFrame({
        'smiles': ['C(#Cc1c2c(nc3ccccc13)CCCCC2)CCN1CCCCC1', 
                   'C(#Cc1cccc(CN2CCOCC2)c1)CCN1CCCCC1', 
                   'C(=C/c1ccccc1)\CN1CCN(C(c2ccccc2)c2ccccc2)CC1'],
        'compound_id': ['CHEMBL66660', 
                        'CHEMBL237087', 
                        'CHEMBL43064'],
    })

    params = Namespace(
        smiles_col='smiles',
        id_col='compound_id',
        moe_threads=1,
    )

    moe_df = feat.compute_all_moe_descriptors(
        smiles_df=smiles_df,
        params=params)

    assert moe_df is not None

def test_compute_all_mordred_descrs_returns_none_on_valueerror(caplog, monkeypatch):
    # Fake "res_df" returned by calc.pandas(...)
    fake_df = mock.MagicMock()
    fake_df.fill_missing.side_effect = ValueError("boom")

    # Fake calculator
    fake_calc = mock.MagicMock()
    fake_calc.pandas.return_value = fake_df

    # Patch the calculator factory used inside the function
    monkeypatch.setattr(feat, "get_mordred_calculator", lambda ignore_3D: fake_calc)

    with caplog.at_level("ERROR"):
        out = feat.compute_all_mordred_descrs(mols=[object()])

    assert out is None
    assert "Mordred descriptor calculation failed in MordredDataFrame.fill_missing" in caplog.text

def test_not_implemented():
    params = param_parser.wrapper({"dataset_key": "fake.csv"})
    f = feat.Featurization(params)
    with pytest.raises(NotImplementedError):
        f.get_feature_count()

    fp = feat.PersistentFeaturization(params)
    with pytest.raises(NotImplementedError):
        fp.featurize_data(
            dset_df=None,
            params=None,
            contains_responses=False
        )


def test_scale_by_heavyatomcount_uses_specified_heavy_atom_col(monkeypatch):
    params = param_parser.wrapper({
        "dataset_key": "fake.csv", 
        "featurizer": "compouted_descriptors",
        "descriptor_type": "rdkit_raw",})
    obj = feat.ComputedDescriptorFeaturization(params)

    # Force a small, known descriptor set for the test
    monkeypatch.setattr(
        obj.__class__,
        "desc_type_cols",
        {"demo": ["MolWt_per_heavyatom", "B_log_scaled", "C"]},
        raising=False,
    )

    df = pd.DataFrame(
        {
            "id": [1, 2],
            "HA": [10, 5],               # specified heavy atom column
            "MolWt": [100.0, 50.0],      # unscaled input for _per_heavyatom
            "B": [1.0, np.e],            # unscaled input for _log_scaled
            "C": [7.0, 9.0],             # pass-through
        }
    )

    out = obj.scale_by_heavyatomcount_and_log_scale(df, descr_type="demo", heavy_atom_col="HA")

    # Columns exist (do not assert full column order, nondesc_cols order is set-derived)
    assert {"id", "HA", "MolWt_per_heavyatom", "B_log_scaled", "C"}.issubset(out.columns)

    # Values are correct
    assert np.allclose(out["MolWt_per_heavyatom"].to_numpy(), np.array([10.0, 10.0]))
    assert np.allclose(out["B_log_scaled"].to_numpy(), np.array([0.0, 1.0]))
    assert np.allclose(out["C"].to_numpy(), df["C"].to_numpy())
    assert np.allclose(out["HA"].to_numpy(), df["HA"].to_numpy())
    assert np.allclose(out["id"].to_numpy(), df["id"].to_numpy())


def test_scale_by_heavyatomcount_raises_runtimeerror_when_cannot_infer_heavy_atom_col(monkeypatch):
    params = param_parser.wrapper({
        "dataset_key": "fake.csv", 
        "featurizer": "compouted_descriptors",
        "descriptor_type": "rdkit_raw",})
    obj = feat.ComputedDescriptorFeaturization(params)

    # No "HeavyAtomCount" or "nHeavyAtom" in descr_cols, and we do not pass heavy_atom_col
    monkeypatch.setattr(
        obj.__class__,
        "desc_type_cols",
        {"demo": ["MolWt_per_heavyatom"]},
        raising=False,
    )

    df = pd.DataFrame({"MolWt": [100.0]})

    with pytest.raises(RuntimeError, match=r"Could not infer heavy atom column"):
        obj.scale_by_heavyatomcount_and_log_scale(df, descr_type="demo")

def test_scale_by_heavyatomcount_and_log_scale_raises_on_negative_log_feature(monkeypatch):
    params = param_parser.wrapper({
        "dataset_key": "fake.csv", 
        "featurizer": "compouted_descriptors",
        "descriptor_type": "rdkit_raw",})
    obj = feat.ComputedDescriptorFeaturization(params)

    # No "HeavyAtomCount" or "nHeavyAtom" in descr_cols, and we do not pass heavy_atom_col
    monkeypatch.setattr(
        obj.__class__,
        "desc_type_cols",
        {"test_type": ["SomeFeature_log_scaled", "HeavyAtomCount"]},
        raising=False,
    )

    # Build a DataFrame where:
    #   - HeavyAtomCount is present so the function can infer heavy atom counts
    #   - SomeFeature has a negative value, and desc_type_cols includes SomeFeature_log_scaled
    df = pd.DataFrame({
        "HeavyAtomCount": [10, 12],
        "SomeFeature": [-1.0, 2.0]  # negative value triggers ValueError in log scaling
    })

    with pytest.raises(ValueError) as excinfo:
        obj.scale_by_heavyatomcount_and_log_scale(
            desc_df=df,
            descr_type="test_type"  # uses desc_type_cols defined above
        )

    # Optionally check the error message
    assert "Cannot log scale SomeFeature" in str(excinfo.value)

if __name__ == '__main__':
    test_compute_moe()