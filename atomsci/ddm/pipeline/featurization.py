"""Classes providing different methods of featurizing compounds and other data entities"""

import logging
import os
import tempfile
import time

import numpy as np
import deepchem as dc
import pandas as pd

from atomsci.ddm.utils import datastore_functions as dsf
from atomsci.ddm.pipeline import transformations as trans
from atomsci.ddm.pipeline import parameter_parser as pp
from atomsci.ddm.pipeline import model_datasets as md
from atomsci.ddm.pipeline import model_pipeline as mp

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolfiles
from rdkit.Chem import rdmolops
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

subclassed_mordred_classes = ['EState', 'MolecularDistanceEdge']
try:
    from mordred import Calculator, descriptors, get_descriptors_from_module
    from mordred.EState import AtomTypeEState, AggrType
    from mordred.MolecularDistanceEdge import MolecularDistanceEdge
    from mordred import BalabanJ, BertzCT, HydrogenBond, MoeType, RotatableBond, SLogP, TopoPSA
    #rdkit_desc_mods = [BalabanJ, BertzCT, HydrogenBond, MoeType, RotatableBond, SLogP, TopoPSA]
    mordred_supported = True
except ImportError:
    mordred_supported = False

feather_supported = True
try:
    import pyarrow.feather as feather
except (ImportError, AttributeError, ModuleNotFoundError):
    feather_supported = False

# TODO: Commented out for now, because Gomez-Bombarelli autoencoder package doesn't work 
# with the version of Keras that is part of TensorFlow 2.5. Eventually we'll replace this
# with the JT-VAE and/or Wasserstein autoencoders.
#try:
#    from mol_vae_features import MoleculeVAEFeaturizer
#except ImportError:
#    pass

import collections

logging.basicConfig(format='%(asctime)-15s %(message)s')
log = logging.getLogger('ATOM')



# ****************************************************************************************
def make_weights(vals, is_class=False):
    """In the case of multitask learning, we must create weights for each sample labeling
    it with 0 if there is not value or 1 if there is.

    Args:
        vals: numpy array containing nans or empty strings where there are not labels

    Returns:
        vals: numpy array same as input vals, but strings are replaced with 0 (for classification models) or nans (for regression models)
        w: numpy array same shape as out_vals, where w[i,j] = 1 if vals[i,j] is nan else w[i,j] = 0
    """
    out_vals = np.copy(vals)
    # sometimes instead of nan, '' is used for missing values
    if not np.issubdtype(out_vals.dtype, np.number):
        # there might be strings or other objects in this array
        out_vals[out_vals==''] = np.nan
    out_vals = out_vals.astype(float)
    w = np.where(np.isnan(out_vals), 0., 1).astype(float)
    if is_class:
        out_vals = np.where(np.isnan(out_vals), 0., out_vals)
        return out_vals, w
    else:
        return out_vals, w


# ****************************************************************************************
def create_featurization(params):
    """Factory method to create the appropriate type of Featurization object for params.featurizer

    Args:
        params (argparse.Namespace: Object containing the parameter list

    Returns:
        Featurization object of the correct subclass as specified by params.featurizer

    Raises:
        ValueError: If params.featurizer not in ['ecfp','graphconv','molvae','computed_descriptors','descriptors']

    """
    #TODO: Change molvae to generic autoencoder
    if params.featurizer in ('ecfp', 'graphconv', 'molvae') \
            or params.featurizer in pp.featurizer_wl:
        return DynamicFeaturization(params)
    elif params.featurizer == 'embedding':
        return EmbeddingFeaturization(params)
    elif params.featurizer in ('descriptors'):
        return DescriptorFeaturization(params)
    elif params.featurizer in ('computed_descriptors'):
        return ComputedDescriptorFeaturization(params)
    else:
        raise ValueError("Unknown featurization type %s" % params.featurizer)

# ****************************************************************************************
def remove_duplicate_smiles(dset_df, smiles_col='rdkit_smiles'):
    """Remove any rows with duplicate SMILES strings from the given dataset.

    Args:
        dset_df (DataFrame): The dataset table.

        smiles_col (str): The column containing SMILES strings.

    Returns:
        filtered_dset_df (DataFrame): The dataset filtered to remove duplicate SMILES strings.

    """
    log.warning("Duplicate smiles strings: " + str([(item, count) for item, count in collections.Counter(
        dset_df[smiles_col].values.tolist()).items() if count > 1]))
    remove = dset_df.duplicated(subset=smiles_col, keep=False)
    dset_df = dset_df[~remove]
    log.warning("All rows with duplicate smiles strings have been removed")
    return dset_df

# ****************************************************************************************
def get_dataset_attributes(dset_df, params):
    """Construct a table mapping compound IDs to SMILES strings and possibly other attributes
    (e.g., dates) specified in params.

    Args:
        dset_df (DataFrame): The dataset table

        params (Namespace): Parsed parameters. The id_col and smiles_col parameters are used
        to specify the columns in dset_df containing compound IDs and SMILES strings, respectively. If
        the parameter date_col is not None, it is used to specify a column of datetime strings associated
        with each compound.

    Returns:
        attr_df (DataFrame): A table of SMILES strings and (optionally) other attributes, indexed by
        compound_id.
    """
    attr_df = pd.DataFrame({
        params.smiles_col: dset_df[params.smiles_col].values},
        index=dset_df[params.id_col])
    if params.date_col is not None:
        attr_df[params.date_col] = [np.datetime64(d) for d in dset_df[params.date_col].values]
    return attr_df

# ****************************************************************************************
def featurize_smiles(df, featurizer, smiles_col, log_every_N=1000):
    """Replacement for DeepChem 2.1 featurize_smiles_df function, which is buggy. Computes
    features using featurizer for dataframe df column given by smiles_col. Returns them as
    a numpy array, along with an array 'is_valid' indicating which rows of the input
    dataframe yielded valid features.
    """
    smiles_strs = df[smiles_col].values

    features = []
    for ind, smiles in enumerate(smiles_strs):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            new_order = rdmolfiles.CanonicalRankAtoms(mol)
            mol = rdmolops.RenumberAtoms(mol, new_order)
        if ind % log_every_N == 0:
            log.info("Featurizing sample %d" % ind)
        features.append(featurizer.featurize([mol]))
    is_valid = np.array([1 if elt.size > 0 else 0 for elt in features], dtype=bool)
    feat_array = np.array([elt for (is_valid, elt) in zip(is_valid, features) if is_valid])
    if len(feat_array.shape) > 1:
        feat_array = np.squeeze(feat_array, axis=1)
    else:
        log.debug("featurize_smiles() produced 1-D array of size %d" % feat_array.shape[0])
        log.debug("Input SMILES had length %d" % len(smiles_strs))
        log.debug("Number of valid SMILES is %d" % sum(is_valid))
    return feat_array, is_valid


# ****************************************************************************************
# Module-level functions for RDKit and Mordred descriptor calculations
# ****************************************************************************************


# ****************************************************************************************
def get_2d_mols(smiles_strs):
    """Convert SMILES strings to RDKit Mol objects without explicit hydrogens or 3D coordinates

    Args:
        smiles_strs (iterable of str): List of SMILES strings to convert

    Returns:
        tuple (mols, is_valid):
            mols (ndarray of Mol): Mol objects for valid SMILES strings only
            is_valid (ndarray of bool): True for each input SMILES string that was valid according to RDKit

    """
    log.debug('Converting SMILES to RDKit Mols')
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_strs]
    is_valid = np.array([(m is not None) for m in mols], dtype=bool)
    mols = np.array(mols)[is_valid]
    return mols, is_valid

def get_3d_mols(smiles_strs):
    """Convert SMILES strings to Mol objects with explicit hydrogens and 3D coordinates

    Args:
        smiles_strs (iterable of str): List of SMILES strings to convert

    Returns:
        tuple (mols, is_valid):
            mols (ndarray of Mol): Mol objects for valid SMILES strings only
            is_valid (ndarray of bool): True for each input SMILES string that was valid according to RDKit

    """
    log.debug('Converting SMILES to RDKit Mols')
    nsmiles = len(smiles_strs)
    mols = [None]*nsmiles
    for i, smi in enumerate(smiles_strs):
        try:
            mols[i] = Chem.MolFromSmiles(smi)
        except TypeError:
            pass

    log.debug('Adding hydrogens to mols')
    mols = [Chem.AddHs(m) if m is not None else None for m in mols]
    log.debug('Computing 3D coordinates')
    for i, m in enumerate(mols):
        if m is not None:
            try:
                AllChem.EmbedMolecule(m)
            except RuntimeError:
                # This sometimes fails in the RDKit code. Give up on this molecule.
                mols[i] = None
    is_valid = np.array([(m is not None) for m in mols], dtype=bool)
    mols = np.array(mols)[is_valid]
    return mols, is_valid


def compute_2d_mordred_descrs(mols):
    """Compute 2D Mordred descriptors only

    Args:
        mols: List of RDKit mol objects for molecules to compute descriptors for.

    Returns:
        res_df: DataFrame containing Mordred descriptors for molecules.

    """
    calc = get_mordred_calculator(ignore_3D=True)
    res_df = calc.pandas(mols)
    try:
        res_df = res_df.fill_missing().applymap(float)
        return res_df
    except ValueError:
        log.error('Mordred descriptor calculation failed in MordredDataFrame.fill_missing')
        return None

def compute_all_mordred_descrs(mols, max_cpus=None, quiet=True):
    """Compute all Mordred descriptors, including 3D ones

    Args:
        mols: List of RDKit mol objects for molecules to compute descriptors for.

        max_cpus: Max number of cores to use for computing descriptors. None means use all available cores.

        quiet: If True, avoid displaying progress indicators for computations.

    Returns:
        res_df: DataFrame containing Mordred descriptors for molecules.

    """
    calc = get_mordred_calculator(ignore_3D=False)
    log.debug("Computing Mordred descriptors")
    res_df = calc.pandas(mols, quiet=quiet, nproc=max_cpus)
    log.debug("Done computing Mordred descriptors")
    try:
        res_df = res_df.fill_missing().applymap(float)
        return res_df
    except ValueError:
        log.error('Mordred descriptor calculation failed in MordredDataFrame.fill_missing')
        return None

def compute_mordred_descriptors_from_smiles(smiles_strs, max_cpus=None, quiet=True, smiles_col='rdkit_smiles'):
    """Compute 2D and 3D Mordred descriptors for the given list of SMILES strings.

    Args:
        smiles_strs:    A list or array of SMILES strings

        max_cpus:       The maximum number of cores to use for computing descriptors. The default value None means
                        that all available cores will be used.
        quiet (bool):   If True, suppress displaying a progress indicator while computing descriptors.

        smiles_col (str): The name of the column that will contain SMILES strings in the returned data frame.

    Returns: tuple
        desc_df (DataFrame): A table of Mordred descriptors for the input SMILES strings that were valid
                        (according to RDKit), together with those SMILES strings.

        is_valid (ndarray of bool): An array of length the same as smiles_strs, indicating which SMILES strings
                            were considered valid.

    """
    mols3d, is_valid = get_3d_mols(smiles_strs)
    if not np.any(is_valid):
        log.error("No valid SMILES strings input to compute_mordred_descriptors_from_smiles.")
        log.error("First input SMILES = %s" % smiles_strs[0])
        return None, is_valid
    desc_df = compute_all_mordred_descrs(mols3d, max_cpus, quiet=quiet)
    valid_smiles = np.array(smiles_strs)[is_valid]
    if desc_df is not None:
        desc_df[smiles_col] = valid_smiles
    else:
        log.debug('Descriptor calculation failed for one of the following SMILES strings:')
        for smi in valid_smiles:
            log.debug(smi)
    return desc_df, is_valid


def compute_all_rdkit_descrs(mol_df, mol_col = "mol"):
    """Compute all RDKit descriptors

    Args:
        mols: List of RDKit Mol objects to compute descriptors for.

    Returns:
        res_df (DataFrame): Data frame containing computed descriptors.

    """
    all_desc = [x[0] for x in Descriptors._descList]
    calc = get_rdkit_calculator(all_desc)
    cds=[]
    for mol in mol_df[mol_col]:
        cd = calc.CalcDescriptors(mol)
        cds.append(cd)
    df2=pd.DataFrame(cds, columns=all_desc, index=mol_df.index)
    mol_df=mol_df.join(df2, lsuffix='', rsuffix='_rdk')
    return mol_df
    

def compute_rdkit_descriptors_from_smiles(smiles_strs, smiles_col='rdkit_smiles'):
    """Compute 2D and 3D RDKit descriptors for the given list of SMILES strings.

    Args:
        smiles_strs:    A list or array of SMILES strings

        max_cpus:       The maximum number of cores to use for computing descriptors. The default value None means
                        that all available cores will be used.
        quiet (bool):   If True, suppress displaying a progress indicator while computing descriptors.

        smiles_col (str): The name of the column that will contain SMILES strings in the returned data frame.

    Returns: tuple
        desc_df (DataFrame): A table of Mordred descriptors for the input SMILES strings that were valid
                        (according to RDKit), together with those SMILES strings.

        is_valid (ndarray of bool): An array of length the same as smiles_strs, indicating which SMILES strings
                            were considered valid.

    """
    mols3d, is_valid = get_3d_mols(smiles_strs)
    mol_df = pd.DataFrame({"mol": mols3d})
    desc_df = compute_all_rdkit_descrs(mol_df)
    valid_smiles = np.array(smiles_strs)[is_valid]
    if desc_df is not None:
        desc_df[smiles_col] = valid_smiles
    else:
        log.debug('Descriptor calculation failed for one of the following SMILES strings:')
        for smi in valid_smiles:
            log.debug(smi)
    return desc_df, is_valid

def get_mordred_calculator(exclude=subclassed_mordred_classes, ignore_3D=False):
    """Create a Mordred calculator with all descriptor modules registered except those whose names are in the exclude list.
    Register ATOM versions of the classes in those modules instead.

    Args:
        exclude (list): List of Mordred descriptor modules to exclude.

        ignore_3D (bool): Whether to exclude descriptors that require computing 3D structures.

    Returns:
        calc (mordred.Calculator): Object for performing Mordred descriptor calculations.

    """
    calc = Calculator(ignore_3D=ignore_3D)
    exclude = ['mordred.%s' % mod for mod in exclude]
    for desc_mod in descriptors.all:
        if desc_mod.__name__ not in exclude:
            calc.register(desc_mod, ignore_3D=ignore_3D)
    calc.register(ATOMAtomTypeEState)
    calc.register(ATOMMolecularDistanceEdge)
    return calc


def get_rdkit_calculator(desc_list):
    """Create a Mordred calculator with only the RDKit wrapper descriptor modules registered"""
    #calc = Calculator(ignore_3D=True)
    #for desc_mod in rdkit_desc_mods:
    #    calc.register(desc_mod, ignore_3D=True)
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_list)
    return calc


# ****************************************************************************************
# Module-level functions for MOE descriptor calculations
# ****************************************************************************************
def compute_all_moe_descriptors(smiles_df, params):
    """Run MOE to compute all 317 standard descriptors.

    Args:
        smiles_df (DataFrame): Table containing SMILES strings and compound IDs

        params (Namespace): Parsed model parameters, used to identify SMILES and compund ID columns.

    Returns:
        descr_df (DataFrame): Table containing the input SMILES strings and compound IDs, the "washed" SMILES
        string prepared by MOE, a sequence index, and columns for each MOE descriptor.
    """

    # TODO: Get MOE_PATH from params
    moe_path = os.environ.get('MOE_PATH', '/usr/workspace/atom/moe2022_site/bin')
    if not os.path.exists(moe_path):
        raise Exception("MOE is not available, or MOE_PATH environment variable needs to be set.")
    moe_root = os.path.abspath('%s/..' % moe_path)
    # Make sure we have an environment variable that points to the license server
    if os.environ.get('LM_LICENSE_FILE', None) is None:
        os.environ['LM_LICENSE_FILE'] = '7002@bribe.llnl.gov'

    moe_args = []
    moe_args.append("{moePath}/moebatch".format(moePath=moe_path))
    moe_args.append("-mpu")
    if params.moe_threads < 0:
        # By default use all available cores but one, times 2 for hyperthreading
        moe_threads = 2 * len(os.sched_getaffinity(0)) - 2
    else:
        moe_threads = params.moe_threads
    moe_args.append('%d' % moe_threads)

    moe_args.append("-exec")
    # TODO: Directory with svl scripts should be part of AMPL installation. The code below is specific to the LC environment.
    moe_template = """db_Close db_Open['{fileMDB}','create']; db_ImportASCII[ascii_file: '{smilesFile}',
    db_file: '{fileMDB}',delimiter: ',', quotes: 0, names: ['original_smiles','cmpd_id'],types: ['char','char']];
    run ['{moeRoot}/custom/ksm_svl/smp_WashMinimizeSMILES.svl', ['{fileMDB}', 'original_smiles']];
    run ['{moeRoot}/custom/svl/db_desc_smp5.svl',['{fileMDB}','mol_prep', [], [codeset: 'All_No_MOPAC_Protein']]];
    dir_export_ASCIIBB ['{fileMDB}',[quotes:1,titles:1]];"""

    #with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = tempfile.mkdtemp()
    curdir = os.getcwd()
    if True:
        # Write SMILES strings and compound IDs to a temp file
        smiles_file = '%s/smiles4moe.csv' % tmpdir
        file_mdb = 'smiles4moe.mdb'
        smiles_df.to_csv(smiles_file, index=False, columns=[params.smiles_col, params.id_col])
        log.debug("Wrote SMILES strings to %s" % smiles_file)
        os.chdir(tmpdir)
        moe_cmds = '"' + moe_template.format(moeRoot=moe_root, smilesFile=smiles_file, fileMDB=file_mdb) + '"'
        moe_args.append(moe_cmds)
        moe_args.append("-exit")
        log.debug('Computing MOE descriptors')
        command = " ".join(moe_args)
        log.debug('Command: %s' % command)
        try:
            shellcmd = '%s >& %s/moe_err.txt' % (command, tmpdir)
            retcode = os.system(shellcmd)
            log.debug('MOE descriptor calculation done')
            log.debug("Return status: %d" % retcode)
            errbuf = open('%s/moe_err.txt' % tmpdir, 'r').read()
            log.debug("\nStderr:\n%s" % errbuf)
            output_file = '%s/smiles4moe.txt' % tmpdir
            if not os.path.exists(output_file):
                log.error('MOE descriptor calculation failed.')
                return None
            log.debug("Reading descriptors from %s" % output_file)
            result_df = pd.read_csv(output_file, index_col=False,
                            dtype={'cmpd_id':'string'}) # IDs should always be treated as strings
            result_df = result_df.rename(columns={'cmpd_id' : params.id_col, 'original_smiles' : params.smiles_col})
            os.chdir(curdir)
            return result_df
        except Exception as e:
            log.error('Failed to invoke MOE to compute descriptors: %s' % str(e))
            os.chdir(curdir)
            raise


# ****************************************************************************************
# ****************************************************************************************
# Featurization classes
# ****************************************************************************************


class Featurization(object):
    """Abstract base class for featurization code

    Attributes:
        feat_type (str): Type of featurizer, set in __init__

    """
    def __init__(self, params):
        """Initializes a Featurization object.

        Args:
            params (Namespace): Contains parameters used to instantiate the featurizer.
        Side effects:
            sets self.feat_type as a Featurization attribute
        """

        self.feat_type = params.featurizer

    # ****************************************************************************************
    def featurize_data(self, dset_df, params, contains_responses):
        """Perform featurization on the given dataset.

        Args:
            dset_df (DataFrame): A table of data to be featurized. At minimum, should include columns.
            for the compound ID and assay value; for some featurizers, must also contain a SMILES string column.

            params (Namespace): Parsed parameters to be used for featurization.

            contains_responses (bool): Whether the dataset being featurized contains response columns.

        Raises:
            NotImplementedError: Must be implemented by concrete subclasses

        """
        # Must be implemented by concrete subclasses
        raise NotImplementedError

    # ****************************************************************************************
    def extract_prefeaturized_data(self, featurized_dset_df, params):
        """Extracts dataset features, values, IDs and attributes from the given prefeaturized data frame.
        Args:
            featurized_dset_df (DataFrame): Data frame for the dataset.

            params (Namespace): Parsed parameters for model.

        Raises:
            NotImplementedError: Must be implemented by concrete subclasses
        """
        raise NotImplementedError

    # ****************************************************************************************
    def get_feature_columns(self):
        """Returns a list of feature column names associated with this Featurization instance.

        Args:
            None

        Raises:
            NotImplementedError: Must be implemented by concrete subclasses
        """
        raise NotImplementedError

    # ****************************************************************************************
    def get_feature_count(self):
        """Returns the number of feature columns associated with this Featurization instance.

        Args:
            None

        Raises:
            NotImplementedError: Must be implemented by concrete subclasses
        """
        raise NotImplementedError

    # ****************************************************************************************
    def create_feature_transformer(self, dataset):
        """Fit a scaling and centering transformation to the feature matrix of the given dataset, and return a
        DeepChem transformer object holding its parameters.

        Args:
            dataset (deepchem.Dataset): featurized dataset

        Returns:
            Empty list

        Raises:
            NotImplementedError: Must be implemented by concrete subclasses
        """
        return []

    # ****************************************************************************************
    def get_feature_specific_metadata(self, params):
        """Returns a dictionary of parameter settings for this Featurization object that are specific
        to the feature type.

        Args:
            params (Namespace): Contains parameters used to instantiate the featurizer.

        """
        raise NotImplementedError

    # ****************************************************************************************
    def get_featurized_dset_name(self, dataset_name):
        """Returns a name for the featurized dataset, for use in filenames or dataset keys.
        Does not include any path information. May be derived from dataset_name.

        Args:
            dataset_name (str): Name of the dataset

        Raises:
            NotImplementedError: Must be implemented by concrete subclasses
        """
        raise NotImplementedError


    # ****************************************************************************************
    def get_featurized_data_subdir(self):
        """Returns the name of a subdirectory (without any leading path information) in which to store
        featurized data, if it goes to the filesystem.

        Raises:
            NotImplementedError: Must be implemented by concrete subclasses
        """
        raise NotImplementedError


# ****************************************************************************************

class DynamicFeaturization(Featurization):
    """Featurization subclass that supports on-the-fly featurization. Can be used when it is inexpensive to
    compute the features. Most DeepChem featurizers are handled through this class.

    Attributes:
        Set in __init__
            feat_type (str): Type of featurizer in ['ecfp','graphconv','molvae']
            featurization_obj: The DeepChem or MoleculeVAEFeaturizer object as determined by feat_type and params
    """
    def __init__(self, params):
        """Initializes a DynamicFeaturization object.

        Args:
            params (Namespace): Contains parameters to be used to instantiate a featurizer.

        Raises:
            ValueError if feat_type not in ['ecfp','graphconv','molvae'], featurization not supported

        Side effects:
            Sets the following DynamicFeaturization attributes:
                feat_type (str): Type of featurizer in ['ecfp','graphconv','molvae']
                featurization_obj: The DeepChem or MoleculeVAEFeaturizer object as determined by feat_type and params
        """

        super().__init__(params)
        if self.feat_type == 'ecfp':
            self.featurizer_obj = dc.feat.CircularFingerprint(size=params.ecfp_size, radius=params.ecfp_radius)
        elif self.feat_type == 'graphconv':
            self.featurizer_obj = dc.feat.ConvMolFeaturizer()
        elif self.feat_type in pp.featurizer_wl:
            kwargs = pp.extract_featurizer_params(params)
            self.featurizer_obj = pp.featurizer_wl[self.feat_type](**kwargs)
        elif self.feat_type == 'embedding':
            # EmbeddingFeaturization doesn't map directly to a DeepChem featurizer
            self.featurizer_obj = None

        #TODO: MoleculeVAEFeaturizer is not working currently. Will be replaced by JT-VAE and cWAE
        # featurizers eventually.
        #elif self.feat_type == 'molvae':
        #    self.featurizer_obj = MoleculeVAEFeaturizer(params.mol_vae_model_file)
        else:
            raise ValueError("Unknown featurization type %s" % self.feat_type)

    # ****************************************************************************************
    def __str__(self):
        """Returns a human-readable description of this Featurization object.

        Returns:
            (str): Describes the featurization type
        """
        return "DynamicFeaturization with %s features" % self.feat_type

    # ****************************************************************************************
    def featurize(self,mols) :
        """Calls DeepChem featurize() object"""

        return self.featurizer_obj.featurize(mols)

    # ****************************************************************************************
    def extract_prefeaturized_data(self, featurized_dset_df, params):
        """Attempts to extract prefeaturized data for the given dataset. For dynamic featurizers, we don't save
        this data, so this method always returns None.

        Args:
            featurized_dset_df (DataFrame): dataset merged with the featurizers

            params (Namespace): Parsed parameters for model.

        Returns:
            None, None, None, None
        """
        return None, None, None, None

    # ****************************************************************************************
    def featurize_data(self, dset_df, params, contains_responses):
        """Perform featurization on the given dataset.

        Args:
            dset_df (DataFrame): A table of data to be featurized. At minimum, should include columns
            for the compound ID and assay value; for some featurizers, must also contain a SMILES string column.

            params (Namespace): Parsed parameters to be used for featurization.

            contains_responses (bool): Whether the dataset being featurized contains response columns.

        Returns:
            Tuple of (features, ids, vals, attr, weights, featurized_dset_df):

            features (np.array): Feature matrix.

            ids (pd.DataFrame): compound IDs or SMILES strings if needed for splitting.

            vals (np.array): array of response values.

            attr (pd.DataFrame): dataframe containing SMILES strings indexed by compound IDs.

            weights (np.array): Array of compound weights

            featurized_dset_df (pd.DataFrame): Data frame combining id, SMILES and response columns from input data frame
            with generated feature columns.

        """
        attr = get_dataset_attributes(dset_df, params)
        features, is_valid = featurize_smiles(dset_df, featurizer=self.featurizer_obj, smiles_col=params.smiles_col)
        if features is None:
            raise Exception("Featurization failed for dataset")
        # Some SMILES strings may not be featurizable. This filters for only valid IDs.

        nrows = sum(is_valid)
        ncols = len(params.response_cols)

        # Select columns to include from the input dataset in the featurized dataset data frame
        dset_cols = [params.id_col, params.smiles_col]
        if contains_responses:
            dset_cols += params.response_cols
        if params.date_col is not None:
            dset_cols.append(params.date_col)
        keep_df = dset_df[dset_cols][is_valid].reset_index(drop=True)
        if len(features.shape) > 1:
            feat_df = pd.DataFrame(features, columns=[f"c{i}" for i in range(features.shape[1])])
        else:
            feat_df = pd.DataFrame(dict(c0=features))
        featurized_dset_df = pd.concat([keep_df, feat_df], ignore_index=False, axis=1)

        is_class=params.model_type=='classification'
        if contains_responses and (params.model_type != 'hybrid'):
            vals, w = make_weights(featurized_dset_df[params.response_cols].values, is_class=is_class) #, self.id_field)
        else:
            vals = np.zeros((nrows,ncols))
            w = np.ones((nrows,ncols)) ## JEA
        attr = attr[is_valid]
        ids = featurized_dset_df[params.id_col]

        assert len(features) == len(ids) == len(vals) == len(w) ## JEA
        return features, ids, vals, attr, w, featurized_dset_df

    # ****************************************************************************************
    def create_feature_transformer(self, dataset):
        """Fit a scaling and centering transformation to the feature matrix of the given dataset, and return a
        DeepChem transformer object holding its parameters.

        Args:
            dataset (deepchem.Dataset): featurized dataset

        Returns:
            Empty list since we will not be transforming the features of a DynamicFeaturization object
        """
        # Dynamic features such as ECFP fingerprints are not typically scaled and centered. This behavior
        # can be overridden if necessary by specific subclasses.
        return []

    # ****************************************************************************************
    def get_featurized_dset_name(self, dataset_name):
        """Returns a name for the featurized dataset, for use in filenames or dataset keys.
        Does not include any path information. May be derived from dataset_name.

        Args:
            dataset_name (str): Name of the dataset

        Raises:
            Exception: This method is not supported by the DynamicFeaturization subclass
        """
        raise NotImplementedError("DynamicFeaturization doesn't support get_featurized_dset_name()")


    # ****************************************************************************************
    def get_featurized_data_subdir(self):
        """Returns the name of a subdirectory (without any leading path information) in which to store
        featurized data, if it goes to the filesystem.

        Raises:
            Exception: This method is not supported by the DynamicFeaturization subclass
        """
        raise Exception("DynamicFeaturization doesn't support get_featurized_data_subdir()")


    # ****************************************************************************************
    def get_feature_columns(self):
        """Returns a list of feature column names associated with this Featurization instance.
        For DynamicFeaturization, the column names are essentially meaningless, so these will
        be "c0, c1, ... etc.".

        Args:
            None

        Returns:
            (list): List of column names in the format ['c0','c1', ...] of the length of the features
        """
        return ['c%d' % i for i in range(self.get_feature_count())]

    # ****************************************************************************************
    def get_feature_count(self):
        """Returns the number of feature columns associated with this Featurization instance.

        Args:
            None

        Returns:
            (int): The number of feature columns for the DynamicFeaturization subclass, feat_type specific
        """
        if self.feat_type == 'ecfp':
            return self.featurizer_obj.size
        elif self.feat_type == 'graphconv':
            return self.featurizer_obj.feature_length()
        elif self.feat_type == 'molvae':
            return self.featurizer_obj.latent_rep_size
        elif self.feat_type == 'MolGraphConvFeaturizer':
            # https://deepchem.readthedocs.io/en/latest/api_reference/featurizers.html#molgraphconvfeaturizer
            # 30 atom features plus 11 edge features
            return 44
        elif self.feat_type in pp.featurizer_wl:
            # It's kind of hard to count some of these features
            return None

    # ****************************************************************************************
    def get_feature_specific_metadata(self, params):
        """Returns a dictionary of parameter settings for this Featurization object that are specific
        to the feature type.

        Args:
            params (Namespace): Argparse Namespace object containing the parameter list

        Returns
            dict: Dictionary containing featurizer specific metadata as a subdict under the keys
            ['ecfp_specific','autoencoder_specific']
        """
        feat_metadata = {}
        # MJT: I changed params.featurizer in this instance to self.feat_type to be syntactically consistent
        if self.feat_type == 'ecfp':
            ecfp_params = dict(ecfp_radius = params.ecfp_radius,
                               ecfp_size = params.ecfp_size)
            feat_metadata['ecfp_specific'] = ecfp_params
        elif self.feat_type == 'graphconv':
            # No graph conv specific params at present
            pass
        elif self.feat_type == 'molvae':
            # TODO: If the parameter name for the model file changes to 'autoencoder_model_key', change it below.
            mol_vae_params = {'autoencoder_model_key': params.mol_vae_model_file}
            feat_metadata['autoencoder_specific'] = mol_vae_params
        elif self.feat_type in pp.featurizer_wl:
            feat_metadata['auto_featurizer_specific'] = pp.extract_featurizer_params(params, strip_prefix=False)
        return feat_metadata

# ****************************************************************************************
class EmbeddingFeaturization(DynamicFeaturization):
    """Featurizer that uses a pretrained AMPL neural network model to compute features consisting
    of the activations from the "embedding" layer of the network. For this to work, the underlying
    DeepChem model must implement the predict_embedding function.
    """

    def __init__(self, params):
        """Initializes an EmbeddingFeaturization object.

        Args:
            params (Namespace): Contains parameters to be used to instantiate the featurizer.

        Raises:
            ValueError if neither embedding_model_uuid or embedding_model_path is specified in params.

        Side effects:
            Sets the following EmbeddingFeaturization attributes:
                embedding_pipeline: A ModelPipeline object for the embedding model.
        """
        super().__init__(params)
        log_level = log.getEffectiveLevel()
        if params.embedding_model_path is not None:
            self.embedding_pipeline = mp.create_prediction_pipeline_from_file(params, reload_dir=None,
                                        model_path=params.embedding_model_path)
        elif params.embedding_model_uuid is not None:
            self.embedding_pipeline = mp.create_prediction_pipeline(params, params.embedding_model_uuid,
                                        collection_name=params.embedding_model_collection)
        else:
            raise ValueError("EmbeddingFeaturizer: must specify either embedding_model_uuid or embedding_model_path")
        # Restore the logging level, which may have been changed by the create_prediction_pipeline function
        log.setLevel(log_level)


    # ****************************************************************************************
    def __str__(self):
        """Returns a human-readable description of this Featurization object.

        Returns:
            (str): Describes the featurization type
        """
        return f"EmbeddingFeaturization based on model UUID {self.embedding_pipeline.model_uuid}"

    # ****************************************************************************************
    def featurize(self,mols) :
        """Calls DeepChem featurize() object"""

        # TODO Does this actually get called externally?
        raise NotImplementedError

    # ****************************************************************************************
    def featurize_data(self, dset_df, params, contains_responses):
        """Perform featurization on the given dataset.

        Args:
            dset_df (DataFrame): A table of data to be featurized. At minimum, should include columns
            for the compound ID and SMILES string

            params (Namespace): Parsed parameters to be used for featurization.

            contains_responses (bool): Whether the dataset being featurized contains response columns.

        Returns:
            Tuple of (features, ids, vals, attr, weights, featurized_dset_df):

            features (np.array): Feature matrix.

            ids (pd.DataFrame): Compound IDs or SMILES strings if needed for splitting.

            attr (pd.DataFrame): Data frame containing SMILES strings indexed by compound IDs.

            vals (np.array): Array of response values.

            weights (np.array): Array of compound weights

            featurized_dset_df (pd.DataFrame): Data frame combining id, SMILES and response columns from input data frame
            with generated feature columns.
        """

        # First featurize the molecules in dset_df using the featurizer of the embedding model. 

        input_featurization = self.embedding_pipeline.model_wrapper.featurization
        self.embedding_pipeline.featurization = input_featurization

        input_model_dataset = md.create_minimal_dataset(self.embedding_pipeline.params,
                                    input_featurization, contains_responses=True)

        input_dset_df = dset_df.copy()
        if contains_responses:
            colmap = {}
            for orig_col, embed_col in zip(params.response_cols, self.embedding_pipeline.params.response_cols):
                colmap[orig_col] = embed_col
            input_dset_df = input_dset_df.rename(columns=colmap)
        input_model_dataset.get_featurized_data(input_dset_df)
        input_dataset = input_model_dataset.dataset
        input_features = input_dataset.X
        ids = input_dataset.ids
        vals = input_dataset.y
        weights = input_dataset.w
        attr = input_model_dataset.attr

        input_dataset = self.embedding_pipeline.model_wrapper.transform_dataset(input_dataset)

        # Run the embedding model to generate features. 
        embedding = self.embedding_pipeline.model_wrapper.generate_embeddings(input_dataset)

        # The DeepChem predict_embedding methods pad their outputs to multiples of the batch size.
        # Strip off this extra padding.
        nrows = input_features.shape[0]
        embedding = embedding[:nrows,:]

        # Select columns to include from the input dataset in the featurized dataset data frame
        dset_cols = [params.id_col, params.smiles_col]
        if contains_responses:
            dset_cols += params.response_cols
        if params.date_col is not None:
            dset_cols.append(params.date_col)
        keep_df = dset_df[dset_cols]
        feat_df = pd.DataFrame(embedding, columns=[f"c{i}" for i in range(embedding.shape[1])])
        featurized_dset_df = pd.concat([keep_df, feat_df], ignore_index=False, axis=1)

        return embedding, ids, vals, attr, weights, featurized_dset_df

    # ****************************************************************************************
    def get_feature_count(self):
        """Returns the number of feature columns associated with this Featurization instance.

        Args:
            None

        Returns:
            (int): The number of feature columns for the DynamicFeaturization subclass, feat_type specific
        """
        # For GraphConv models, the embedding layer is a GraphGather layer that effectively doubles the number 
        # of nodes in the final Dense layer, which is given by the last element of params.layer_sizes.
        # For other NN models, the embedding layer has the number of nodes specified by that last element.
        if self.embedding_pipeline.params.featurizer == 'graphconv':
            return 2*self.embedding_pipeline.params.layer_sizes[-1]
        else:
            return self.embedding_pipeline.params.layer_sizes[-1]


    # ****************************************************************************************
    def get_feature_specific_metadata(self, params):
        """Returns a dictionary of parameter settings for this Featurization object that are specific
        to the feature type.

        Args:
            params (Namespace): Argparse Namespace object containing the parameter list

        Returns
            dict: Dictionary containing featurizer specific metadata as a subdict under the key
            'embedding_specific'.
        """
        feat_metadata = {}

        embedding_params = dict(
            embedding_model_uuid = params.embedding_model_uuid,
            embedding_model_collection = params.embedding_model_collection,
            embedding_model_path = params.embedding_model_path)
        feat_metadata['embedding_specific'] = embedding_params

        return feat_metadata


# ****************************************************************************************

class PersistentFeaturization(Featurization):
    """Subclass for featurizers that support persistent storage of featurized data. Used when computing or mapping
    the features is CPU- or memory-intensive, e.g. descriptors. Currently DescriptorFeaturization is the only subclass,
    but others are planned (e.g., UMAPDescriptorFeaturization).
    """
    def __init__(self, params):
        """Initializes a PersistentFeaturization object. This is a good place to load data used by the featurizer,
        such as a table of descriptors.

        Args:
            params (Namespace): Contains parameters to be used to instantiate a featurizer.
        """
        super().__init__(params)

    # ****************************************************************************************
    def extract_prefeaturized_data(self, featurized_dset_df, params):
        """Attempts to extract prefeaturized data for the given dataset.

        Args:
            featurized_dset_df (DataFrame): dataset merged with the featurizers

            params (Namespace): Parsed parameters for model.

        Raises:
            NotImplementedError: Currently, only DescriptorFeaturization is supported, is not a generic method
        """
        # TODO: Is it possible to implement this generically for all persistent featurizers?
        raise NotImplementedError

    # ****************************************************************************************
    def featurize_data(self, dset_df, params, contains_responses):
        """Perform featurization on the given dataset.

        Args:
            dset_df (DataFrame): A table of data to be featurized. At minimum, should include columns.
            for the compound ID and assay value; for some featurizers, must also contain a SMILES string column.

            params (Namespace): Parsed parameters to be used for featurization.

            contains_responses (bool): Whether the dataset being featurized contains response columns.

        Returns:
            Tuple of (features, ids, vals, attr, weights, featurized_dset_df):

            features (np.array): Feature matrix.

            ids (pd.DataFrame): Compound IDs or SMILES strings if needed for splitting.

            attr (pd.DataFrame): Data frame containing SMILES strings indexed by compound IDs.

            vals (np.array): Array of response values.

            weights (np.array): Array of compound weights

            featurized_dset_df (pd.DataFrame): Data frame combining id, SMILES and response columns from input data frame
            with generated feature columns.

        Raises:
            NotImplementedError: Implemented by subclasses.
        """
        raise NotImplementedError

    # ****************************************************************************************
    def create_feature_transformer(self, dataset):
        """Fit a scaling and centering transformation to the feature matrix of the given dataset, and return a
        DeepChem transformer object holding its parameters.

        Args:
            dataset (deepchem.Dataset): featurized dataset

        """
        # Leave it to subclasses to determine if features should be scaled and centered.
        return []

# ****************************************************************************************

class DescriptorFeaturization(PersistentFeaturization):
    """Subclass for featurizers that map sets of (usually) precomputed descriptors to compound IDs; the resulting merged
    dataset is persisted to the filesystem or datastore.
    
    Attributes:
        Set in __init_:
        feat_type (str): Type of featurizer, set in super.(__init__)

        descriptor_type (str): The type of descriptor

        descriptor_key (str): The path to the descriptor featurization matrix if it saved to a file,
        or the key to the file in the Datastore

        descriptor_base (str/path): The base path to the descriptor featurization matrix

        precomp_descr_table (pd.DataFrame): initialized as an empty DataFrame, will be overridden to contain the
        full descriptor table

    Class attributes:
        supported_descriptor_types
        all_desc_col
    """

    supported_descriptor_types = []
    desc_type_cols = {}
    desc_type_scaled = {}
    desc_type_source = {}

    # ****************************************************************************************
    # (ksm): Made this a class method. A DescriptorFeaturization instance only supports
    # one descriptor_type, so making the list of supported descriptor types an instance attribute
    # was misleading.

    @classmethod
    def load_descriptor_spec(cls, desc_spec_bucket, desc_spec_key) :
        """Read a descriptor specification table from the datastore or the filesystem.
        The table is a CSV file with the following columns:

        descr_type:     A string specifying a descriptor source/program and a subset of descriptor columns

        source:         Name of the program/package that generates the descriptors

        scaled:         Binary indicator for whether subset of descriptor values are scaled by molecule's atom count

        descriptors:    A semicolon separated list of descriptor columns.

        The values in the table are used to set class variables desc_type_cols, desc_type_source and desc_type_scaled.

        Args:
            desc_spec_bucket : bucket where descriptor spec is located

            desc_spec_key: data store key, or full file path to locate descriptor spec object

        Returns:
            None

        Side effects:
            Sets the following class variables:

            cls.desc_type_cols -> map from decriptor types to their associated descriptor column names

            cls.desc_type_source -> map from decriptor types to the program/package that generates them

            cls.desc_type_scaled -> map from decriptor types to boolean indicators of whether some descriptor
            values are scaled.

            cls.supported_descriptor_types  -> the list of available descriptor types
        """

        if desc_spec_bucket == '':
            ds_client = None
        else:
            try:
                ds_client = dsf.config_client()
            except Exception:
                ds_client = None
        cls.desc_type_cols = {}
        cls.desc_type_scaled = {}
        cls.desc_type_source = {}

        # If a datastore client is not detected or a datastore bucket is not specified
        # assume that the ds_key is a full path pointer to a file on the file system. If that
        # file doesn't exist, use the fallback file installed with the AMPL package.
        desc_spec_key_fallback = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                              'data', 'descriptor_sets_sources_by_descr_type.csv')

        if ds_client is None or desc_spec_bucket == '':
            if os.path.exists(desc_spec_key):
                log.debug("Reading descriptor spec table from %s" % desc_spec_key)
                desc_spec_df = pd.read_csv(desc_spec_key, index_col=False)
            else:
                log.debug("Reading descriptor spec table from %s" % desc_spec_key_fallback)
                desc_spec_df = pd.read_csv(desc_spec_key_fallback, index_col=False)
        else :
            # Try the descriptor_spec_key parameter first, then fall back to package file
            try:
                desc_spec_df = dsf.retrieve_dataset_by_datasetkey(desc_spec_key, desc_spec_bucket, ds_client)
            except:
                desc_spec_df = pd.read_csv(desc_spec_key_fallback, index_col=False)

        for desc_type, source, scaled, descriptors in zip(desc_spec_df.descr_type.values,
                                                          desc_spec_df.source.values,
                                                          desc_spec_df.scaled.values,
                                                          desc_spec_df.descriptors.values):
            cls.desc_type_cols[desc_type] = descriptors.split(';')
            cls.desc_type_source[desc_type] = source
            cls.desc_type_scaled[desc_type] = bool(scaled)

        cls.supported_descriptor_types = list(cls.desc_type_source.keys())

    def __init__(self, params):
        """Initializes a DescriptorFeaturization object. This is a good place to load data used by the featurizer,
        such as a table of descriptors.

        Args:
            params (Namespace): Contains parameters to be used to instantiate a featurizer.

        Side effects:
            Sets the following attributes of DescriptorFeaturization:

            feat_type (str): Type of featurizer, set in __init__

            descriptor_type (str): The type of descriptor

            descriptor_key (str): The path to the precomputed descriptor table if it is saved to a file, or
            the key to the file in the Datastore

            descriptor_base (str/path): The base name of the precomputed descriptor table file, without the
            directory and extension

            precomp_descr_table (pd.DataFrame): The precomputed descriptor table itself. Initialized as an empty
            DataFrame, will be replaced later on first call to featurize_data().

            desc_id_col (str): Name of the column in precomp_descr_table containing compound IDs

            desc_smiles_col (str): Name of the column in precomp_descr_table, if any, containing compound SMILES
        """
        super().__init__(params)
        cls = self.__class__
        # Load mapping between descriptor types and lists of descriptors
        if not params.datastore:
            params.descriptor_spec_bucket = ''
        if len(cls.supported_descriptor_types) == 0:
            cls.load_descriptor_spec(params.descriptor_spec_bucket, params.descriptor_spec_key)

        if params.descriptor_type not in cls.supported_descriptor_types:
            raise ValueError("Unsupported descriptor type %s" % params.descriptor_type)
        self.descriptor_type = params.descriptor_type
        self.descriptor_key = params.descriptor_key
        if self.descriptor_key is not None:
            self.descriptor_base = os.path.splitext(os.path.basename(params.descriptor_key))[0]
        else:
            self.descriptor_base = None
        self.desc_id_col = None
        self.desc_smiles_col = None


        # Load an empty descriptor table. We'll load the real table later the first time we need it.
        self.precomp_descr_table = pd.DataFrame()

    # ****************************************************************************************
    def __str__(self):
        """Returns a human-readable description of this Featurization object.
        Returns:
            (str): Describes the featurization type
        """
        return "DescriptorFeaturization with %s descriptors" % self.descriptor_type


    # ****************************************************************************************
    def extract_prefeaturized_data(self, featurized_dset_df, params):
        """Attempts to retrieve prefeaturized data for the given dataset.

        Args:
            featurized_dset_df (pd.DataFrame): dataset merged with the featurizers

            params (Namespace): Parsed parameters for model.

        Returns:
            Tuple of (features, ids, vals, attr).

            features (np.array): Feature matrix.

            ids (pd.DataFrame): compound IDs or SMILES strings if needed for splitting.

            attr (pd.DataFrame): dataframe containing SMILES strings indexed by compound IDs.

            vals (np.array): array of response values.

        """
        md.check_task_columns(params, featurized_dset_df)
        user_specified_features = self.get_feature_columns()
        featurizer_obj = dc.feat.UserDefinedFeaturizer(user_specified_features)
        features = get_user_specified_features(featurized_dset_df, featurizer=featurizer_obj,
                                                                   verbose=False)
        features = features.astype(float)
        ids = featurized_dset_df[params.id_col]
        vals = featurized_dset_df[params.response_cols].values
        attr = get_dataset_attributes(featurized_dset_df, params)

        return features, ids, vals, attr

    # ****************************************************************************************
    def load_descriptor_table(self, params):
        """Load the table of precomputed feature values for the descriptor type specified in params, from
        the datastore_key or path specified by params.descriptor_key and params.descriptor_bucket. Will try
        to load the table from the local filesystem if possible, but the table should at least have a
        metadata record in the datastore. The local file path is the same as descriptor_key on twintron-blue,
        and may be taken from the LC_path metadata property if it is set.

        Args:
            params (Namespace): Parameters for the current pipeline instance.

        Returns:
            None

        Side effects:
            Overwrites the attribute precomp_descr_table (pd.DataFrame) with the loaded descriptor table.
            Sets attributes desc_id_col and desc_smiles_col, if possible, based on the datastore metadata for the
            descriptor table. Otherwise, sets them to reasonable defaults. Note that not all descriptor tables
            contain SMILES strings, but one is required if the table is to be used with ComputedDescriptorFeaturization.
        """

        # Check if we have a datastore client to work with
        if params.datastore:
            try:
                ds_client = dsf.config_client()
            except Exception as e:
                log.warning('Exception when trying to connect to the datastore:')
                log.warning(e)
                ds_client = None
        else:
            ds_client = None
        file_type = ''
        local_path = self.descriptor_key
        if ds_client != None :
            # First get the datastore metadata for the descriptor table. Ideally this will exist even if the table
            # itself lives in the filesystem.
            desc_metadata = dsf.retrieve_dataset_by_datasetkey(self.descriptor_key, bucket=params.descriptor_bucket,
                                                               client=ds_client, return_metadata=True)
            file_type = desc_metadata['distribution']['dataType']
            kv_dict = dsf.get_key_val(desc_metadata['metadata'])
            self.desc_id_col = kv_dict.get('id_col', self.desc_id_col)
            self.desc_smiles_col = kv_dict.get('smiles_col', self.desc_smiles_col)

            # Some descriptor tables in datastore have a metadata property LC_path that point to where
            # the table can be found locally on LC systems. Use this if we're running on LC.
            lc_path = kv_dict.get('LC_path', local_path)
            # Loading from the datastore is much slower than loading from disk, so check if file exists locally
            # and download it otherwise
            if params.system == 'LC' :
                local_path = lc_path
            if self.precomp_descr_table.empty and not os.path.exists(local_path):
                log.info("Reading descriptor table from datastore key=%s bucket=%s" %
                         (self.descriptor_key, params.descriptor_bucket))
                self.precomp_descr_table = dsf.retrieve_dataset_by_datasetkey(self.descriptor_key,
                                                                              params.descriptor_bucket, ds_client)
                log.info("Done reading descriptor table from datastore")

        if self.precomp_descr_table.empty:
            log.info("Loading descriptor table from %s" % local_path)
            if local_path.endswith('.csv') or (file_type != '' and file_type == 'csv') :
                ## DeepChem's transformer complained that the elements were not float, if I don't cast them as such here
                ## not sure why this is happening (JEA)
                dtype_map=dict((el,np.float64) for el in self.get_feature_columns())
                self.precomp_descr_table = pd.read_csv( open(local_path, mode='rt'),dtype=dtype_map)
            elif local_path.endswith('.feather') or (file_type != '' and file_type == 'feather'):
                if not feather_supported:
                    raise Exception("feather package not installed in current environment")
                self.precomp_descr_table = feather.read_dataframe(local_path)
            else:
                raise ValueError("Unknown descriptor table file format: %s" % local_path)
            log.info("Done loading descriptor table from filesystem.")

        # If ID column not in metadata, see if descriptor table has one of the same name as the
        # dataset ID column, or failing that, a reasonable default.
        id_choices = [params.id_col, 'compound_id']
        if self.desc_id_col is None:
            for id_col in id_choices:
                if id_col in self.precomp_descr_table.columns.values:
                    self.desc_id_col = id_col
                    break

        # If SMILES column not in metadata, see if descriptor table has one of the same name as the
        # dataset SMILES column, or else a reasonable default.
        smiles_choices = [params.smiles_col, 'rdkit_smiles', 'base_rdkit_smiles', 'smiles', 'SMILES']
        if self.desc_smiles_col is None:
            for smiles_col in smiles_choices:
                if smiles_col in self.precomp_descr_table.columns.values:
                    self.desc_smiles_col = smiles_col
                    break
        if self.desc_smiles_col is None:
            raise Exception(f"No SMILES column found for descriptor table {self.descriptor_key}")


    # ****************************************************************************************
    def featurize_data(self, dset_df, params, contains_responses):
        """Perform featurization on the given dataset.

        Args:
            dset_df (DataFrame): A table of data to be featurized. At minimum, should include columns.
            for the compound ID and assay value; for some featurizers, must also contain a SMILES string column.

            params (Namespace): Parsed parameters to be used for featurization.

            contains_responses (bool): Whether the dataset being featurized contains response columns.

        Returns:
            Tuple of (features, ids, vals, attr, weights, featurized_dset_df):

            features (np.array): Feature matrix.

            ids (pd.DataFrame): Compound IDs or SMILES strings if needed for splitting.

            attr (pd.DataFrame): Data frame containing SMILES strings indexed by compound IDs.

            vals (np.array): Array of response values

            weights (np.array): Array of compound weights

            featurized_dset_df (pd.DataFrame): Data frame combining id, SMILES and response columns from input data frame
            with generated feature columns.

        Raises:
            Exception: if features is None, feautirzation failed for the dataset

        Side effects:
            Overwrites the attribute precomp_descr_table (pd.DataFrame) with the appropriate descriptor table
        """
        # Compound ID and SMILES columns will be labeled the same as in the input dataset, unless overridden by
        # properties of the precomputed descriptor table
        self.load_descriptor_table(params)
        if self.desc_id_col is None:
            raise Exception('Unable to find compound ID column in descriptor table %s' % params.descriptor_key)

        attr = get_dataset_attributes(dset_df, params)
        dset_cols = [params.id_col]
        if params.date_col is not None:
            dset_cols.append(params.date_col)
        # Include SMILES column from dataset in columns to be merged, unless the descriptor table
        # already has a column of the same name
        if params.smiles_col not in self.precomp_descr_table.columns.values:
            dset_cols.append(params.smiles_col)
        if contains_responses:
            dset_cols += params.response_cols
        featurized_dset_df = dset_df[dset_cols].merge(
                self.precomp_descr_table, how='left', left_on=params.id_col, right_on=self.desc_id_col)

        user_specified_features = self.get_feature_columns()

        featurizer_obj = dc.feat.UserDefinedFeaturizer(user_specified_features)
        features = get_user_specified_features(featurized_dset_df, featurizer=featurizer_obj,
                                                                   verbose=False)
        if features is None:
            raise Exception("Featurization failed for dataset")

        ids = featurized_dset_df[params.id_col]

        nrows = len(ids)
        ncols = len(params.response_cols)
        is_class= params.model_type=='classification'
        if contains_responses and (params.model_type != 'hybrid'):
            vals = featurized_dset_df[params.response_cols].values
            vals, weights = make_weights(vals, is_class=is_class)
        else:
            vals = np.zeros((nrows,ncols))
            weights = np.ones_like(vals, dtype=np.float32)

        attr = attr.loc[ids]

        return features, ids, vals, attr, weights, featurized_dset_df

    # ****************************************************************************************
    def get_featurized_dset_name(self, dataset_name):
        """Returns a name for the featurized dataset, for use in filenames or dataset keys.
        Does not include any path information. May be derived from dataset_name.

        Args:
            dataset_name (str): Name of the dataset

        Returns:
            (str): A name for the feauturized dataset
        """
        return 'subset_%s_%s.csv' % (self.descriptor_base, dataset_name)


    # ****************************************************************************************
    def get_featurized_data_subdir(self):
        """Returns the name of a subdirectory (without any leading path information) in which to store
        featurized data, if it goes to the filesystem.

        Returns:
            (str): 'scaled_descriptors'
        """
        return 'scaled_descriptors'


    # ****************************************************************************************
    def get_feature_columns(self):
        """Returns a list of feature column names associated with this Featurization instance.

        Args:
            None

        Returns:
            (list): List of column names of the features, pulled from DescriptorFeaturization attributes

        """

        return self.__class__.desc_type_cols[self.descriptor_type ]

    # ****************************************************************************************
    def get_feature_count(self):
        """Returns the number of feature columns associated with this Featurization instance.

        Args:
            None

        Returns:
            (int): Number of feature columns associated with DescriptorFeaturization

        """
        return len(self.get_feature_columns())

    # ****************************************************************************************
    def create_feature_transformer(self, dataset):
        """Fit a scaling and centering transformation to the feature matrix of the given dataset, and return a
        DeepChem transformer object holding its parameters.

        Args:
            dataset (deepchem.Dataset): featurized dataset

        Returns:
            (list of DeepChem transformer objects): list of transformers for the feature matrix
        """
        transformers_x = [trans.NormalizationTransformerMissingData(transform_X=True, dataset=dataset)]
        return transformers_x


    # ****************************************************************************************
    def get_feature_specific_metadata(self, params):
        """Returns a dictionary of parameter settings for this Featurization object that are specific
        to the feature type.

        Args:
            params (Namespace): Argparse Namespace argument containing the parameters
        """
        feat_metadata = {}
        desc_params = dict(descriptor_type = self.descriptor_type,
                           descriptor_key = self.descriptor_key,
                           descriptor_bucket = params.descriptor_bucket)
        feat_metadata['descriptor_specific'] = desc_params
        return feat_metadata

# ****************************************************************************************

class ComputedDescriptorFeaturization(DescriptorFeaturization):
    """Subclass for featurizers that support online computation of descriptors, usually given SMILES
    strings or RDKit Mol objects as input rather than compound IDs. The computed descriptors may be cached
    and combined with tables of precomputed descriptors to speed up access. Featurized datasets may be
    persisted to the filesystem or datastore.

    Attributes:
        Set in __init_:

        feat_type (str): Type of featurizer, set in super.(__init__)

        descriptor_type (str): The type of descriptor

        descriptor_key (str): The path to the descriptor featurization matrix if it saved to a file, or the key to
        the file in the Datastore

        descriptor_base (str/path): The base path to the descriptor featurization matrix

        precomp_descr_table (pd.DataFrame): initialized as empty df, will be overridden to contain full descriptor table
    """


    def __init__(self, params):
        """Initializes a ComputedDescriptorFeaturization object.

        Args:
            params (Namespace): Contains parameters to be used to instantiate a featurizer.

        Side effects:
            Sets the following attributes of DescriptorFeaturization

            feat_type (str): Type of featurizer, set in __init__

            descriptor_type (str): The type of descriptor

            descriptor_key (str): The path to the descriptor featurization matrix if it saved to a file,
            or the key to the file in the Datastore

            descriptor_base (str/path): The base path to the descriptor featurization matrix

            precomp_descr_table (pd.DataFrame): initialized as an empty DataFrame, will be overridden to contain
            the full descriptor table
        """
        super().__init__(params)
        cls = self.__class__
        if params.descriptor_type not in cls.supported_descriptor_types:
            raise ValueError("Descriptor type %s is not in the supported descriptor_type list" % params.descriptor_type)



    # ****************************************************************************************
    def featurize_data(self, dset_df, params, contains_responses):
        """Perform featurization on the given dataset, by computing descriptors from SMILES strings or matching them
        to SMILES in precomputed table.

        Args:
            dset_df (DataFrame): A table of data to be featurized. At minimum, should include columns.
            for the compound ID and assay value; for some featurizers, must also contain a SMILES string column.

            params (Namespace): Parsed parameters to be used for featurization.

            contains_responses (bool): Whether the dataset being featurized contains response columns.

        Returns:
            Tuple of (features, ids, vals, attr, weights, featurized_dset_df):

            features (np.array): Feature matrix.

            ids (pd.DataFrame): compound IDs or SMILES strings if needed for splitting.

            attr (pd.DataFrame): dataframe containing SMILES strings indexed by compound IDs.

            vals (np.array): array of response values

            weights (np.array): Array of compound weights

            featurized_dset_df (pd.DataFrame): Data frame combining id, SMILES and response columns from input data frame
            with feature columns.

        Raises:
            Exception: if features is None, featurization failed for the dataset

        Side effects:
            Loads a precomputed descriptor table and sets self.precomp_descr_table to point to it, if one is
            specified by params.descriptor_key.

        """
        use_precomputed = False
        descr_cols = self.get_feature_columns()

        # Try to load a precomputed descriptor table, and check that it's useful to us
        if params.descriptor_key is not None and self.precomp_descr_table.empty:
            self.load_descriptor_table(params)
        if not self.precomp_descr_table.empty:
            if self.desc_smiles_col is None or self.desc_id_col is None:
                log.warning("Precomputed descriptor table %s lacks a SMILES column and/or an ID column." % params.descriptor_key)
                log.warning("Will compute all descriptors on the fly.")
                self.precomp_descr_table = pd.DataFrame()
            else:
                # Check that descriptor table provides all the columns required by the current descriptor_type.
                # If not, it's of no use to us.
                absent_cols = sorted(set(descr_cols) - set(self.precomp_descr_table.columns.values))
                if len(absent_cols) > 0:
                    log.warning("Precomputed descriptor table %s lacks columns needed for descriptor type %s:" % (
                                 params.descriptor_key, params.descriptor_type))
                    log.warning(", ".join(absent_cols))
                    log.warning("Will compute all descriptors on the fly.")
                else:
                    # Descriptor table is good. Reduce it to only the columns we need.
                    self.precomp_descr_table = self.precomp_descr_table[[self.desc_id_col,self.desc_smiles_col]+descr_cols]
                    use_precomputed = True

        # If we were unable to determine which descriptor table columns to use for the compound ID or SMILES string,
        # then we're computing all descriptors. Use the same column names as were specified for the input dataset
        # for whichever columns are missing.
        if self.desc_id_col is None:
            self.desc_id_col = params.id_col
        if self.desc_smiles_col is None:
            self.desc_smiles_col = params.smiles_col

        # Select columns to include from the input dataset in the featurized dataset
        dset_cols = [params.id_col, params.smiles_col]
        if params.date_col is not None:
            dset_cols.append(params.date_col)
        if contains_responses:
            dset_cols += params.response_cols
        keep_df = dset_df[dset_cols]

        # Make a table with duplicate SMILES removed. Later we'll join the descriptors back to the original
        # table.
        input_df = keep_df.drop_duplicates(subset=params.smiles_col)[[params.id_col, params.smiles_col]]

        # Identify which SMILES strings in the dataset need to have descriptors calculated for them and
        # which already have them precomputed
        dset_smiles = set(input_df[params.smiles_col].values)
        if use_precomputed:
            calc_smiles = dset_smiles - set(self.precomp_descr_table[self.desc_smiles_col].values)
            precomp_smiles = dset_smiles - calc_smiles
        else:
            calc_smiles = dset_smiles
            precomp_smiles = set()

        # Compute descriptors for the compounds that need them
        if len(calc_smiles) > 0:
            calc_smiles_df = input_df[input_df[params.smiles_col].isin(calc_smiles)]
            calc_desc_df, is_valid = self.compute_descriptors(calc_smiles_df, params)
            calc_smiles_feat_df = calc_smiles_df[is_valid].reset_index(drop=True)[[params.smiles_col]]
            # update descr_cols with smiles col to merge instead of concat data
            df_cols=[self.desc_smiles_col]
            df_cols.extend(descr_cols)
            calc_smiles_feat_df=calc_smiles_feat_df.merge(calc_desc_df[df_cols], how='left', on=self.desc_smiles_col)
            
            if len(precomp_smiles) == 0:
                uniq_smiles_feat_df = calc_smiles_feat_df

            # Add the newly computed descriptors to the precomputed table
            new_desc_df = calc_desc_df.rename(columns={params.smiles_col : self.desc_smiles_col,
                                                       params.id_col : self.desc_id_col})
            if self.precomp_descr_table.empty:
                self.precomp_descr_table = new_desc_df
            else:
                self.precomp_descr_table = pd.concat([self.precomp_descr_table, new_desc_df], ignore_index=True)

        # Merge descriptors from the precomputed table for the remaining compounds
        if len(precomp_smiles) > 0:
            precomp_smiles_feat_df = self.precomp_descr_table[self.precomp_descr_table[self.desc_smiles_col].isin(
                                                                                                    precomp_smiles)].copy()
            if (params.smiles_col in precomp_smiles_feat_df.columns.values):
                psf_df = precomp_smiles_feat_df
            else:
                psf_df = precomp_smiles_feat_df.rename(columns={self.desc_smiles_col : params.smiles_col})

            # Remove duplicate SMILES matches
            psf_df = psf_df.drop_duplicates(subset=params.smiles_col)

            # Get rid of any extra columns
            psf_df = psf_df[[params.smiles_col]+descr_cols]

            if len(calc_smiles) == 0:
                uniq_smiles_feat_df = psf_df

        # Combine the computed and precomputed data frames, if we had both
        if len(precomp_smiles) > 0 and len(calc_smiles) > 0:
            uniq_smiles_feat_df = pd.concat([calc_smiles_feat_df, psf_df], ignore_index=True)

        # Merge descriptors with input data on SMILES strings.
        featurized_dset_df = keep_df.merge(uniq_smiles_feat_df, how='left', on=params.smiles_col)
        
        # Shuffle the order of rows, so that compounds with precomputed descriptors are intermixed with those having
        # newly computed descriptors. This avoids bias later when doing scaffold splits; otherwise test set will be
        # biased toward non-precomputed compounds.
        featurized_dset_df = featurized_dset_df.sample(frac=1.0)

        # Use the DeepChem featurizer to construct the feature array
        featurizer_obj = dc.feat.UserDefinedFeaturizer(descr_cols)
        features = get_user_specified_features(featurized_dset_df, featurizer=featurizer_obj,
                                                                   verbose=False)
        if features is None:
            raise Exception("UserDefinedFeaturizer failed for dataset")

        # Construct the other components of a DeepChem Dataset object
        ids = featurized_dset_df[params.id_col]
        nrows = len(ids)
        ncols = len(params.response_cols)
        is_class=params.model_type=='classification'
        if contains_responses and (params.model_type != 'hybrid'):
            vals = featurized_dset_df[params.response_cols].values
            vals, weights = make_weights(vals, is_class=is_class)
        else:
            vals = np.zeros((nrows,ncols))
            weights = np.ones_like(vals, dtype=np.float32)

        # Create a table of SMILES strings and other attributes indexed by compound IDs
        attr = get_dataset_attributes(featurized_dset_df, params)

        return features, ids, vals, attr, weights, featurized_dset_df

    # ****************************************************************************************
    def get_featurized_dset_name(self, dataset_name):
        """Returns a name for the featurized dataset, for use in filenames or dataset keys.
        Does not include any path information. May be derived from dataset_name.

        Args:
            dataset_name (str): Name of the dataset

        Returns:
            (str): A name for the feauturized dataset

        """
        return '%s_with_%s_descriptors.csv' % (dataset_name, self.descriptor_type)


    # ****************************************************************************************
    def compute_descriptors(self, smiles_df, params):
        """Compute descriptors for the SMILES strings given in smiles_df.

        Args:
            smiles_df: DataFrame containing SMILES strings to compute descriptors for.

            params (Namespace): Argparse Namespace argument containing the parameters

        Returns:
            ret_df (DataFrame): Data frame containing the compound IDs, SMILES string and descriptor columns as
            specified in the current parameters.

        """
        cls = self.__class__
        descr_source = cls.desc_type_source[params.descriptor_type]
        descr_cols = cls.desc_type_cols[params.descriptor_type]
        descr_scaled = cls.desc_type_scaled[params.descriptor_type]

        if descr_source == 'mordred':
            if not mordred_supported:
                raise Exception("mordred package needs to be installed to use Mordred descriptors")
            desc_df, is_valid = self.compute_mordred_descriptors(smiles_df[params.smiles_col].values, params)
            desc_df = desc_df[descr_cols]
            # Add the ID and SMILES columns to the returned data frame
            ret_df = smiles_df[is_valid][[params.id_col, params.smiles_col]].reset_index(drop=True)
            ret_df = ret_df.rename(columns={params.id_col : self.desc_id_col,
                                        params.smiles_col : self.desc_smiles_col})
            desc_df = desc_df.reset_index(drop=True)
            ret_df = ret_df.join(desc_df, how='inner')

        elif descr_source == 'rdkit':
            desc_df, is_valid = self.compute_rdkit_descriptors(smiles_df, smiles_col = params.smiles_col)
            desc_df = desc_df[descr_cols]
            # Add the ID and SMILES columns to the returned data frame
            ret_df = smiles_df[is_valid][[params.id_col, params.smiles_col]].reset_index(drop=True)
            ret_df = ret_df.rename(columns={params.id_col : self.desc_id_col,
                                        params.smiles_col : self.desc_smiles_col})
            desc_df = desc_df.reset_index(drop=True)
            ret_df = ret_df.join(desc_df, how='inner')

        elif descr_source == 'moe':
            desc_df, is_valid = self.compute_moe_descriptors(smiles_df, params)
            # Add scaling by a_count if descr_scaled is True
            if descr_scaled:
                ret_df = self.scale_moe_descriptors(desc_df, params.descriptor_type)
            else:
                ret_df = desc_df

        else:
            raise ValueError('Unsupported descriptor_type %s' % params.descriptor_type)

        return ret_df, is_valid


    # ****************************************************************************************
    def compute_mordred_descriptors(self, smiles_strs, params):
        """Compute Mordred descriptors for the given list of SMILES strings

        Args:
            smiles_strs (iterable): SMILES strings to compute descriptors for.

            params (Namespace): Argparse Namespace argument containing the parameters.

        Returns:
            (tuple): Tuple containing:

                desc_df (DataFrame): Data frame containing computed descriptors

                is_valid (ndarray of bool): True for each input SMILES string that was valid according to RDKit

        """
        mols3d, is_valid = get_3d_mols(smiles_strs)
        quiet = not params.verbose
        desc_df = compute_all_mordred_descrs(mols3d, params.mordred_cpus, quiet=quiet)
        return desc_df, is_valid

    # ****************************************************************************************
    def compute_rdkit_descriptors(self, smiles_df, smiles_col="rdkit_smiles"):
        """Compute RDKit descriptors for the given list of SMILES strings

        Args:
            smiles_strs: SMILES strings to compute descriptors for.

        Returns:
            (tuple): Tuple containing:

                desc_df (DataFrame): Data frame containing computed descriptors

                is_valid (ndarray of bool): True for each input SMILES string that was valid according to RDKit

        """
        smiles_df["mol"], is_valid = get_3d_mols(smiles_df[smiles_col])
        desc_df = compute_all_rdkit_descrs(smiles_df, "mol")
        return desc_df, is_valid

    # ****************************************************************************************
    def compute_moe_descriptors(self, smiles_df, params):
        """Compute MOE descriptors for the given list of SMILES strings

        Args:
            smiles_strs (iterable): SMILES strings to compute descriptors for.

            params (Namespace): Argparse Namespace argument containing the parameters.

        Returns:
            (tuple): Tuple containing:

                desc_df (DataFrame): Data frame containing computed descriptors

                is_valid (ndarray of bool): True for each input SMILES string that was valid according to RDKit

        """
        nsmiles = smiles_df.shape[0]
        desc_df = compute_all_moe_descriptors(smiles_df, params)

        # MOE ignores SMILES strings it can't parse, so we have to mark as invalid the corresponding
        # input compounds
        desc_ids = set(desc_df[params.id_col].values)
        is_valid = np.array([id in desc_ids for id in smiles_df[params.id_col].values])
        num_invalid = len(is_valid) - sum(is_valid)
        if num_invalid > 0:
            log.warning("MOE did not compute descriptors for %d/%d SMILES strings" % (num_invalid, nsmiles))
        return desc_df, is_valid

    # ****************************************************************************************
    def scale_moe_descriptors(self, desc_df, descr_type):
        """Scale selected descriptors computed by MOE by dividing their values by the atom count per molecule.

        Args:
            desc_df (DataFrame): Data frame containing computed descriptors.

            descr_type (str): Descriptor type, used to look up expected set of descriptor columns.

        Returns:
            scaled_df (DataFrame): Data frame with scaled descriptors.

        """
        cls = self.__class__
        descr_cols = cls.desc_type_cols[descr_type]
        a_count = desc_df.a_count.values
        unscaled_moe_desc_cols = [col.replace('_per_atom', '') for col in descr_cols]
        nondesc_cols = list(set(desc_df.columns.values) - set(unscaled_moe_desc_cols))
        scaled_cols=[desc_df[nondesc_cols].copy()]
        for scaled_col, unscaled_col in zip(descr_cols, unscaled_moe_desc_cols):
            if scaled_col.endswith('_per_atom'):
                tmp_col=desc_df[unscaled_col] / a_count
                tmp_col=tmp_col.rename(scaled_col)
                scaled_cols.append(tmp_col)
            else:
                tmp_col=desc_df[unscaled_col]
                tmp_col=tmp_col.rename(scaled_col)
                scaled_cols.append(tmp_col)
        scaled_df=pd.concat(scaled_cols, axis=1)
        return scaled_df.copy()

    # ****************************************************************************************
    def __str__(self):
        """Returns a human-readable description of this Featurization object.
        Returns:
            (str): Describes the featurization type
        """
        return "ComputedDescriptorFeaturization with %s descriptors" % self.descriptor_type

# ****************************************************************************************
def get_user_specified_features(df, featurizer, verbose=False):
    """Temp fix for DC 2.3 issue. See
    https://github.com/deepchem/deepchem/issues/1841
    """

    """Extract and merge user specified features.

    Merge features included in dataset provided by user
    into final features dataframe

    Three types of featurization here:

    1) Molecule featurization
    -) Smiles string featurization
    -) Rdkit MOL featurization
    2) Complex featurization
    -) PDB files for interacting molecules.
    3) User specified featurizations.

    """
    time1 = time.time()
    df[featurizer.feature_fields] = df[featurizer.feature_fields].apply(pd.to_numeric)
    X_shard =  df[featurizer.feature_fields].to_numpy()
    time2 = time.time()
    if verbose:
        log.info("TIMING: user specified processing took %0.3f s" % (time2 - time1))
    return X_shard

# **************************************************************************************************************
# Subclasses of Mordred descriptor classes
# **************************************************************************************************************

if mordred_supported:

    class ATOMAtomTypeEState(AtomTypeEState):
        """EState descriptors restricted to those that can be computed for most compounds"""

        my_es_types = ['sCH3','dCH2','ssCH2','dsCH','aaCH','sssCH','tsC','dssC','aasC','aaaC','ssssC','sNH2','ssNH',
                       'aaNH','tN','dsN','aaN','sssN','ddsN','aasN','sOH','dO','ssO','aaO','sF','dS','ssS','aaS',
                       'ddssS','sCl','sBr']


        @classmethod
        def preset(cls, version):
            return (
                cls(a, t) for a in [AggrType.count, AggrType.sum]
                          for t in cls.my_es_types
                )

    class ATOMMolecularDistanceEdge(MolecularDistanceEdge):
        """MolecularDistanceEdge descriptors restricted to those that can be computed for most compounds"""

        @classmethod
        def preset(cls, version):
            return (cls(a, b, 6) for a in [2,3] for b in range(a, 4) )
