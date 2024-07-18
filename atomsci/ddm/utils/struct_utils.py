"""Functions to manipulate and convert between various representations of chemical structures: SMILES, InChi and RDKit Mol objects.
Many of these functions (those with a 'workers' argument) accept either a single SMILES or InChi string or a list of strings
as their first argument, and return a value with the same datatype. If a list is passed and the 'workers' argument is > 1,
the calculation is parallelized across multiple threads; this can save significant time when operating on thousands of
molecules.
"""

import re
import numpy as np
import molvs

from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

stdizer = molvs.standardize.Standardizer(prefer_organic=True)
uncharger = molvs.charge.Uncharger()


def get_rdkit_smiles(orig_smiles, useIsomericSmiles=True):
    """Given a SMILES string, regenerate a "canonical" SMILES string for the same molecule
    using the implementation in RDKit.

    Args:
        orig_smiles (str): SMILES string to canonicalize.

        useIsomericSmiles (bool): Whether to retain stereochemistry information in the generated string.

    Returns:
        str: Canonicalized SMILES string.

    """
    mol = Chem.MolFromSmiles(orig_smiles)
    if mol is None:
        return ""
    else:
        return Chem.MolToSmiles(mol, isomericSmiles=useIsomericSmiles)


def rdkit_smiles_from_smiles(orig_smiles, useIsomericSmiles=True, useCanonicalTautomers=False, workers=1):
    """Parallel version of get_rdkit_smiles. If orig_smiles is a list and workers is > 1, spawn 'workers'
    threads to convert input SMILES strings to standardized RDKit format.

    Args:
        orig_smiles (list or str): List of SMILES strings to canonicalize.

        useIsomericSmiles (bool): Whether to retain stereochemistry information in the generated strings.

        useCanonicalTautomers (bool): Whether to convert the generated SMILES to their canonical tautomers. Defaults
        to False for backward compatibility.

        workers (int): Number of parallel threads to use for calculation.

    Returns:
        list or str: Canonicalized SMILES strings.

    """

    if isinstance(orig_smiles, list):
        from functools import partial
        func = partial(rdkit_smiles_from_smiles, useIsomericSmiles=useIsomericSmiles, 
                       useCanonicalTautomers=useCanonicalTautomers)
        if workers > 1:
            from multiprocessing import pool
            batchsize = 200
            batches = [orig_smiles[i:i + batchsize] for i in range(0, len(orig_smiles), batchsize)]
            with pool.Pool(workers) as p:
                rdkit_smiles = p.map(func, batches)
                rdkit_smiles = [y for x in rdkit_smiles for y in x]  # Flatten results
        else:
            rdkit_smiles = [func(smi) for smi in orig_smiles]
    else:
        # Actual standardization code, everything above here is for multiprocessing and list parsing
        std_mol = Chem.MolFromSmiles(orig_smiles)
        if std_mol is None:
            rdkit_smiles = ""
        else:
            if useCanonicalTautomers:
                taut_enum = rdMolStandardize.TautomerEnumerator()
                std_mol = taut_enum.Canonicalize(std_mol)
            rdkit_smiles = Chem.MolToSmiles(std_mol, isomericSmiles=useIsomericSmiles)
    return rdkit_smiles


def mols_from_smiles(orig_smiles, workers=1):
    """Parallel function to create RDKit Mol objects for a list of SMILES strings. If orig_smiles is a list
    and workers is > 1, spawn 'workers' threads to convert input SMILES strings to Mol objects.

    Args:
        orig_smiles (list or str): List of SMILES strings to convert to Mol objects.

        workers (int): Number of parallel threads to use for calculation.

    Returns:
        list of rdkit.Chem.Mol: RDKit objects representing molecules.

    """

    if isinstance(orig_smiles, list):
        from functools import partial
        func = partial(mols_from_smiles)
        if workers > 1:
            from multiprocessing import pool
            batchsize = 200
            batches = [orig_smiles[i:i + batchsize] for i in range(0, len(orig_smiles), batchsize)]
            with pool.Pool(workers) as p:
                mols = p.map(func, batches)
                mols = [y for x in mols for y in x]  # Flatten results
        else:
            mols = [func(smi) for smi in orig_smiles]
    else:
        # Actual standardization code, everything above here is for multiprocessing and list parsing
        mols = Chem.MolFromSmiles(orig_smiles)
    return mols


def base_smiles_from_smiles(orig_smiles, useIsomericSmiles=True, removeCharges=False, 
                            useCanonicalTautomers=False, workers=1):
    """Generate standardized SMILES strings for the largest fragments of each molecule specified by
    orig_smiles. Strips salt groups and replaces any rare isotopes with the most common ones for each element.

    Args:
        orig_smiles (list or str): List of SMILES strings to canonicalize.

        useIsomericSmiles (bool): Whether to retain stereochemistry information in the generated strings.

        removeCharges (bool): If true, add or remove hydrogens to produce uncharged molecules.

        useCanonicalTautomers (bool): Whether to convert the generated SMILES to their canonical tautomers. Defaults
        to False for backward compatibility.

        workers (int): Number of parallel threads to use for calculation.

    Returns:
        list or str: Canonicalized SMILES strings.

    """

    if isinstance(orig_smiles, list):
        from functools import partial
        func = partial(base_smiles_from_smiles, useIsomericSmiles=useIsomericSmiles, removeCharges=removeCharges,
                       useCanonicalTautomers=useCanonicalTautomers)
        if workers > 1:
            from multiprocessing import pool
            batchsize = 200
            batches = [orig_smiles[i:i + batchsize] for i in range(0, len(orig_smiles), batchsize)]
            with pool.Pool(workers) as p:
                base_smiles = p.map(func, batches)
                base_smiles = [y for x in base_smiles for y in x]  # Flatten results
        else:
            base_smiles = [func(smi) for smi in orig_smiles]
    else:
        # Actual standardization code, everything above here is for multiprocessing and list parsing
        std_mol = base_mol_from_smiles(orig_smiles, useIsomericSmiles, removeCharges)
        if std_mol is None:
            base_smiles = ""
        else:
            if useCanonicalTautomers:
                taut_enum = rdMolStandardize.TautomerEnumerator()
                std_mol = taut_enum.Canonicalize(std_mol)
            base_smiles = Chem.MolToSmiles(std_mol, isomericSmiles=useIsomericSmiles)
    return base_smiles


def kekulize_smiles(orig_smiles, useIsomericSmiles=True, workers=1):
    """Generate Kekulized SMILES strings for the molecules specified by orig_smiles. Kekulized SMILES strings
    are ones in which aromatic rings are represented by uppercase letters with alternating single and
    double bonds, rather than lowercase letters; they are needed by some external applications.

    Args:
        orig_smiles (list or str): List of SMILES strings to Kekulize.

        useIsomericSmiles (bool): Whether to retain stereochemistry information in the generated strings.

        workers (int): Number of parallel threads to use for calculation.

    Returns:
        list or str: Kekulized SMILES strings.

    """

    if isinstance(orig_smiles, list):
        from functools import partial
        func = partial(kekulize_smiles, useIsomericSmiles=useIsomericSmiles)
        if workers > 1:
            from multiprocessing import pool
            batchsize = 200
            batches = [orig_smiles[i:i + batchsize] for i in range(0, len(orig_smiles), batchsize)]
            with pool.Pool(workers) as p:
                kekulized_smiles = p.map(func, batches)
                kekulized_smiles = [y for x in kekulized_smiles for y in x]  # Flatten results
        else:
            kekulized_smiles = [func(smi) for smi in orig_smiles]
    else:
        std_mol = Chem.MolFromSmiles(orig_smiles)
        if std_mol is None:
            kekulized_smiles = ""
        else:
            Chem.Kekulize(std_mol)
            kekulized_smiles = Chem.MolToSmiles(std_mol, kekuleSmiles=True, isomericSmiles=useIsomericSmiles)
    return kekulized_smiles


def base_mol_from_smiles(orig_smiles, useIsomericSmiles=True, removeCharges=False):
    """Generate a standardized RDKit Mol object for the largest fragment of the molecule specified by
    orig_smiles. Replace any rare isotopes with the most common ones for each element.
    If removeCharges is True, add hydrogens as needed to eliminate charges.

    Args:
        orig_smiles (str): SMILES string to standardize.

        useIsomericSmiles (bool): Whether to retain stereochemistry information in the generated string.

        removeCharges (bool): If true, add or remove hydrogens to produce uncharged molecules.

    Returns:
        str: Standardized salt-stripped SMILES string.

    """
    if type(orig_smiles) != str:
        return None
    if len(orig_smiles) == 0:
        return None
    cmpd_mol = Chem.MolFromSmiles(orig_smiles)
    if cmpd_mol is None:
        return None
    std_mol = stdizer.isotope_parent(stdizer.fragment_parent(cmpd_mol), skip_standardize=True)
    if removeCharges:
        std_mol = uncharger(std_mol)
    return std_mol


def base_smiles_from_inchi(inchi_str, useIsomericSmiles=True, removeCharges=False, workers=1):
    """Generate standardized salt-stripped SMILES strings for the largest fragments of each molecule represented by
    InChi string(s) inchi_str. Replaces any rare isotopes with the most common ones for each element.

    Args:
        inchi_str (list or str): List of InChi strings to convert.

        useIsomericSmiles (bool): Whether to retain stereochemistry information in the generated strings.

        removeCharges (bool): If true, add or remove hydrogens to produce uncharged molecules.

        workers (int): Number of parallel threads to use for calculation.

    Returns:
        list or str: Standardized SMILES strings.

    """

    if isinstance(inchi_str, list):
        from functools import partial
        func = partial(base_smiles_from_inchi, useIsomericSmiles=useIsomericSmiles, removeCharges=removeCharges)
        if workers > 1:
            from multiprocessing import pool
            batchsize = 200
            batches = [inchi_str[i:i + batchsize] for i in range(0, len(inchi_str), batchsize)]
            with pool.Pool(workers) as p:
                base_smiles = p.map(func, batches)
                base_smiles = [y for x in base_smiles for y in x]  # Flatten results
        else:
            base_smiles = [func(inchi) for inchi in inchi_str]
    else:
        # Actual standardization code, everything above here is for multiprocessing and list parsing
        std_mol = base_mol_from_inchi(inchi_str, useIsomericSmiles, removeCharges)
        if std_mol is None:
            base_smiles = ""
        else:
            base_smiles = Chem.MolToSmiles(std_mol, isomericSmiles=useIsomericSmiles)
    return base_smiles


def base_mol_from_inchi(inchi_str, useIsomericSmiles=True, removeCharges=False):
    """Generate a standardized RDKit Mol object for the largest fragment of the molecule specified by
    InChi string inchi_str. Replace any rare isotopes with the most common ones for each element.
    If removeCharges is True, add hydrogens as needed to eliminate charges.

    Args:
        inchi_str (str): InChi string representing molecule.

        useIsomericSmiles (bool): Whether to retain stereochemistry information in the generated string.

        removeCharges (bool): If true, add or remove hydrogens to produce uncharged molecules.

    Returns:
        str: Standardized salt-stripped SMILES string.

    """
    if type(inchi_str) != str:
        return None
    if len(inchi_str) == 0:
        return None
    cmpd_mol = Chem.inchi.MolFromInchi(inchi_str)
    if cmpd_mol is None:
        return None
    std_mol = stdizer.isotope_parent(stdizer.fragment_parent(cmpd_mol), skip_standardize=True)
    if removeCharges:
        std_mol = uncharger(std_mol)
    return std_mol


def draw_structure(smiles_str, image_path, image_size=500):
    """Draw structure for the compound with the given SMILES string as a PNG file.

    Note that there are more flexible functions for drawing structures in the rdkit_easy module.
    This function is only retained for backward compatibility.

    Args:
        smiles_str (str): SMILES representation of compound.

        image_path (str): Filepath for image file to be generated.

        image_size (int): Width of square bounding box for image.

    Returns:
        None.

    """
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        print(("Unable to read original SMILES for %s" % cmpd_num))
    else:
        _discard = AllChem.Compute2DCoords(mol)
        Draw.MolToFile(mol, image_path, size=(image_size, image_size), fitImage=False)


def _standardize_chemistry(df, standard='rdkit', smiles_col='rdkit_smiles', workers=1):
    """Function used by merge_dataframes_by_smiles. Converts SMILES strings in a given column of a data frame
    into either standardized salt-stripped SMILES strings or InChi strings.
    """
    smiles = list(df[smiles_col])
    out = []
    if standard.lower() == 'rdkit':
        col = 'rdkit_smiles'
        out = base_smiles_from_smiles(smiles, workers=workers)

    elif standard.lower() == 'inchi':
        col = 'InCHI'
        for smi in smiles:
            try:
                mol = Chem.MolFromSmiles(smi)
                out.append(Chem.inchi.MolToInchi(mol))
            except:
                out.append('Invalid SMILES: %s' % (smi))
    elif std == 'name':
        print('Name technique currently not implemented')
    else:
        raise Exception('Unrecognized standardization type: %s' % (standard))
    df[col] = out

    return df, col


def _merge_values(values, strategy='list'):
    """Function used by merge_dataframes_by_smiles. Returns a summary of the values in 'values', unless
    'strategy' == 'list', in which case it returns values itself.
    """
    try:
        values.remove('')
    except:
        values = values

    if values is None:
        val = float('NaN')
    elif strategy == 'list':
        val = values
    elif strategy == 'uniquelist':
        val = list(set(values))
    elif strategy == 'mean':
        val = np.mean(values)
    elif strategy == 'geomean':
        val = np.geomean(values)
    elif strategy == 'median':
        val = np.median(values)
    elif strategy == 'mode':
        val = np.mode(values)
    elif strategy == 'max':
        val = max(values)
    elif strategy == 'min':
        val = min(values)
    else:
        raise Exception('Unknown column merge strategy: %s', columnmerge)

    if type(val) is list and len(val) == 1:
        val = val[0]
    return val


def _merge_dataframes_by_smiles(dataframes, smiles_col='rdkit_smiles', id_col='compound_id', how='outer', comparetype='rdkit',
                               columnmerge=None, workers=1):
    """Merge two dataframes labeled by SMILEs strings on a rdkit or InCHI canonicalization to identify shared compounds

    DEPRECATED. This function was added by a developer no longer at ATOM and doesn't appear to be used by any
    extant code.
    """

    left_df, joincol = _standardize_chemistry(dataframes[0], standard=comparetype, workers=workers)
    for idx, df in enumerate(dataframes[1:]):
        df, joincol = _standardize_chemistry(df, standard=comparetype, smiles_col='rdkit_smiles', workers=workers)
        new_df = left_df.merge(df, how=how, on=[joincol])
        new_df = new_df.fillna('')
        if columnmerge is not None:
            shared_cols = list(set(left_df.columns.values) & set(df.columns.values))
            shared_cols.remove(joincol)
            for col in shared_cols:
                lCol = col + '_x'
                rCol = col + '_y'
                vals = list(zip(new_df[lCol], new_df[rCol]))
                vals = [list(i) for i in vals]
                if col == id_col:
                    vals = [_merge_values(i, strategy='min') for i in vals]
                else:
                    vals = [_merge_values(i, strategy=columnmerge) for i in vals]

                new_df[col] = vals
                new_df = new_df.drop([lCol, rCol], axis=1)
        left_df = new_df

    return new_df


def smiles_to_inchi_key(smiles):
    """Generates an InChI key from a SMILES string.  Note that an InChI key is different from an InChI *string*;
    it can be used as a unique identifier, but doesn't hold the information needed to reconstruct a molecule.

    Args:
        smiles (str): SMILES string.

    Returns:
        str: An InChI key. Returns None if RDKit cannot convert the SMILES string to an RDKit Mol object.

    """
    m = Chem.MolFromSmiles(smiles)
    if m:
        inchi = Chem.MolToInchi(m)
        inchi_key = Chem.InchiToInchiKey(inchi)
    else:
        inchi_key = None

    return inchi_key


def fix_moe_smiles(smiles):
    """Correct the SMILES strings generated by MOE to standardize the representation of protonated atoms,
    so that RDKit can read them.

    Args:
        smiles (str): SMILES string.

    Returns:
        str: The corrected SMILES string.

    """
    protn_pat = re.compile(r'\[([cCBnNPS])([-\+])(@*)(H[1234]*)*\]')
    scalar = False
    if type(smiles) == str:
        smiles = [smiles]
        scalar = True
    fixed = []
    for smi in smiles:
        fixed.append(protn_pat.sub(r'[\1\3\4\2]', smi))
    if scalar:
        fixed = fixed[0]
    return fixed


def mol_wt_from_smiles(smiles, workers=1):
    """Calculate molecular weights for molecules represented by SMILES strings.

    Args:
        smiles (list or str): List of SMILES strings.

        workers (int): Number of parallel threads to use for calculations.

    Returns:
        list or float: Molecular weights. NaN is returned for SMILES strings that could not be read by RDKit.

    """

    if isinstance(smiles, list):
        from functools import partial
        func = partial(mol_wt_from_smiles)
        if workers > 1:
            from multiprocessing import pool
            batchsize = 200
            batches = [smiles[i:i + batchsize] for i in range(0, len(smiles), batchsize)]
            with pool.Pool(workers) as p:
                mol_wt = p.map(func, batches)
                mol_wt = [y for x in mol_wt for y in x]  # Flatten results
        else:
            mol_wt = [func(smi) for smi in smiles]
    else:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            mol_wt = np.nan
        else:
            mol_wt = Descriptors.MolWt(mol)
    return mol_wt


def canonical_tautomers_from_smiles(smiles):
    """Returns SMILES strings for the canonical tautomers of a SMILES string or list of SMILES strings

    Args:
        smiles (list or str): List of SMILES strings.

    Returns:
        (list of str) : List of SMILES strings for the canonical tautomers.
    """
    taut_enum = rdMolStandardize.TautomerEnumerator()
    if type(smiles) == str:
        smiles = [smiles]
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    canon_tautomers = [taut_enum.Canonicalize(m) if m is not None else None for m in mols]
    return [Chem.MolToSmiles(m) if m is not None else '' for m in canon_tautomers]


