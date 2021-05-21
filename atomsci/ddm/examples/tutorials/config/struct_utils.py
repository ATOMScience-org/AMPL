# Functions to do various things with compound structures

import os
import re
import pdb
import numpy as np
import molvs

from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors

stdizer = molvs.standardize.Standardizer(prefer_organic=True)
uncharger = molvs.charge.Uncharger()


def get_rdkit_smiles(orig_smiles, useIsomericSmiles=True):
  """
  Given a SMILES string, regenerate a "canonical" SMILES string for the same molecule
  using the implementation in RDKit. If useIsomericSmiles is false, stereochemistry information
  will be removed in the generated string.
  """
  mol = Chem.MolFromSmiles(orig_smiles)
  if mol is None:
    return ""
  else:
    return Chem.MolToSmiles(mol, isomericSmiles=useIsomericSmiles)


def base_smiles_from_smiles(orig_smiles, useIsomericSmiles=True, removeCharges=False, workers=1):
  """
  Generate a standardized SMILES string for the largest fragment of the molecule specified by
  orig_smiles. Replace any rare isotopes with the most common ones for each element.
  If removeCharges is True, add hydrogens as needed to eliminate charges.
  """
  
  if isinstance(orig_smiles,list):
    from functools import partial
    func = partial(base_smiles_from_smiles,useIsomericSmiles=useIsomericSmiles,removeCharges=removeCharges)
    if workers > 1:
      from multiprocessing import pool
      batchsize = 200
      batches = [orig_smiles[i:i+batchsize] for i in range(0, len(orig_smiles), batchsize)]
      with pool.Pool(workers) as p:
        base_smiles = p.map(func,batches)
        base_smiles = [y for x in base_smiles for y in x] #Flatten results
    else:
      base_smiles = [func(smi) for smi in orig_smiles]
  else:
    # Actual standardization code, everything above here is for multiprocessing and list parsing
    std_mol = base_mol_from_smiles(orig_smiles, useIsomericSmiles, removeCharges)
    if std_mol is None:
      base_smiles = ""
    else:
      base_smiles = Chem.MolToSmiles(std_mol, isomericSmiles=useIsomericSmiles)
  return base_smiles

def kekulize_smiles(orig_smiles, useIsomericSmiles=True, workers=1):
  """
  Generate a Kekulized SMILES string for the molecule specified by
  orig_smiles. 
  """
  
  if isinstance(orig_smiles,list):
    from functools import partial
    func = partial(kekulize_smiles,useIsomericSmiles=useIsomericSmiles)
    if workers > 1:
      from multiprocessing import pool
      batchsize = 200
      batches = [orig_smiles[i:i+batchsize] for i in range(0, len(orig_smiles), batchsize)]
      with pool.Pool(workers) as p:
        kekulized_smiles = p.map(func,batches)
        kekulized_smiles = [y for x in kekulized_smiles for y in x] #Flatten results
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
  """
  Generate a standardized RDKit Mol object for the largest fragment of the molecule specified by
  orig_smiles. Replace any rare isotopes with the most common ones for each element.
  If removeCharges is True, add hydrogens as needed to eliminate charges.
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
  """
  Generate a standardized SMILES string for the largest fragment of the molecule specified by
  InChi string inchi_str. Replace any rare isotopes with the most common ones for each element.
  If removeCharges is True, add hydrogens as needed to eliminate charges. If useIsomericSmiles
  is True (the default), retain stereochemistry info in the generated SMILES string.
  Note that inchi_str may be a list, in which case a list of SMILES strings is generated.
  If workers > 1 and inchi_str is a list, the calculations are parallelized over the given number
  of worker threads.
  """
  
  if isinstance(inchi_str,list):
    from functools import partial
    func = partial(base_smiles_from_inchi, useIsomericSmiles=useIsomericSmiles, removeCharges=removeCharges)
    if workers > 1:
      from multiprocessing import pool
      batchsize = 200
      batches = [inchi_str[i:i+batchsize] for i in range(0, len(inchi_str), batchsize)]
      with pool.Pool(workers) as p:
        base_smiles = p.map(func,batches)
        base_smiles = [y for x in base_smiles for y in x] #Flatten results
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
  """
  Generate a standardized RDKit Mol object for the largest fragment of the molecule specified by
  InChi string inchi_str. Replace any rare isotopes with the most common ones for each element.
  If removeCharges is True, add hydrogens as needed to eliminate charges. 
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
  """
  Draw structure for the compound with the given SMILES string, in a PNG file
  with the given path.
  """
  mol = Chem.MolFromSmiles(smiles_str)
  if mol is None:
    print(("Unable to read original SMILES for %s" % cmpd_num))
  else:
    _discard = AllChem.Compute2DCoords(mol)
    Draw.MolToFile(mol, image_path, size=(image_size,image_size), fitImage=False)


def standardize_chemistry(df,standard='rdkit',smiles_col='rdkit_smiles',workers=1):

    smiles = list(df[smiles_col])
    out = []
    if standard.lower() == 'rdkit':
        col = 'rdkit_smiles'
        out = base_smiles_from_smiles(smiles,workers=workers)

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

def merge_values(values,strategy='list'):
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
        
def merge_dataframes_by_smiles(dataframes,smiles_col='rdkit_smiles',id_col='compound_id',how='outer',comparetype='rdkit',columnmerge=None,workers=1):
    """
    Merge two dataframes labeled by SMILEs strings on a rdkit or InCHI cannonicalization to identify shared compounds
    """
    
    left_df, joincol = standardize_chemistry(dataframes[0], standard=comparetype, workers=workers)
    for idx,df in enumerate(dataframes[1:]):
        df, joincol = standardize_chemistry(df, standard=comparetype, smiles_col='rdkit_smiles', workers=workers)
        new_df = left_df.merge(df, how=how, on=[joincol])
        new_df = new_df.fillna('')
        if columnmerge is not None:
            shared_cols = list(set(left_df.columns.values) & set(df.columns.values))
            shared_cols.remove(joincol)
            for col in shared_cols:
                lCol = col + '_x'
                rCol = col + '_y'
                vals = list(zip(new_df[lCol],new_df[rCol]))
                vals = [list(i) for i in vals]
                if col == id_col:
                    vals = [merge_values(i,strategy='min') for i in vals]
                else:
                    vals = [merge_values(i,strategy=columnmerge) for i in vals]
                    
                new_df[col] = vals    
                new_df = new_df.drop([lCol,rCol],axis=1)
        left_df = new_df
                    
    return new_df

def smiles_to_inchi_key (smiles):

    """InChI key from SMILES string.  SMILES > RDKit molecule object >
       InChI string > InChI key.
       Returns None if cannot convert SMILES string to RDKit molecule.
    """
    m = Chem.MolFromSmiles (smiles)
    if m:
        inchi = Chem.MolToInchi (m)
        inchi_key = Chem.InchiToInchiKey (inchi)
    else:
        inchi_key = None

    return inchi_key


def fix_moe_smiles(smiles):
    """
    Correct the SMILES strings generated by MOE to standardize the representation of protonated atoms,
    so that RDKit can read them.
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

