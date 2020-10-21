"""
Utilities for clustering and visualizing compound structures using RDKit. 
"""
# Mostly written by Logan Van Ravenswaay, with additions and edits by Ben Madej and Kevin McLoughlin.

import os

from IPython.display import SVG, HTML

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import PandasTools
from rdkit.Chem.Draw import MolToImage, rdMolDraw2D
from rdkit.ML.Cluster import Butina
from rdkit.ML.Descriptors import MoleculeDescriptors


def add_mol_column(df, smiles_col, molecule_col='mol'):
    """
    Add a column 'molecule_col' to data frame 'df' containing RDKit Mol objects
    corresponding to the SMILES strings in column 'smiles_col'.
    """
    PandasTools.AddMoleculeColumnToFrame(df, smiles_col, molecule_col, includeFingerprints=True)
    return df


def calculate_descriptors(df, molecule_column='mol'):
    """
    Uses RDKit to compute various descriptors for compounds in the given data frame. Expects
    compounds to be represented by RDKit Mol objects in the column given by molecule_column.
    Returns the input data frame with added columns for the descriptors.
    """

    descriptors = [x[0] for x in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptors)
    for i in df.index:
        cd = calculator.CalcDescriptors(df.at[i, molecule_column])
        for desc, d in list(zip(descriptors, cd)):
            df.at[i, desc] = d


def cluster_dataframe(df, molecule_column='mol', cluster_column='cluster', cutoff=0.2):
    """
    From RDKit cookbook http://rdkit.org/docs_temp/Cookbook.html. Performs Butina clustering
    on the molecules represented as Mol objects in column molecule_column of data frame df,
    using cutoff as the maximum Tanimoto distance for identifying neighbors of each molecule.
    Returns the input dataframe with an extra column 'cluster_column' containing the cluster
    index for each molecule.
    """
    df[cluster_column] = -1
    df2 = df.reset_index()
    df2['df_index'] = df.index
    mols = df2[[molecule_column]].values.tolist()
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(x[0], 2, 1024) for x in mols]
    clusters = cluster_fingerprints(fingerprints, cutoff=cutoff)
    for i in range(len(clusters)):
        c = clusters[i]
        for j in c:
            df_index = df2.at[j, 'df_index']
            df.at[df_index, cluster_column] = i


def cluster_fingerprints(fps, cutoff=0.2):
    """
    From RDKit cookbook http://rdkit.org/docs_temp/Cookbook.html. Given a list of fingerprint
    bit vectors fps, performs Butina clustering using the given Tanimoto distance cutoff.
    Returns a list of cluster indices for each fingerprint.
    """

    # first generate the distance matrix:
    dists = []
    nfps = len(fps)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1-x for x in sims])

    # now cluster the data:
    cs = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
    return cs


def mol_to_html(mol, name, type='svg', directory='rdkit_svg', width=400, height=200):
    """
    Creates a PNG or SVG image file for the given molecule's structure  with filename 'name' in the given directory.
    Returns an HTML image tag referencing the image file.
    """
    img_file = '%s/%s' % (directory, name)
    os.makedirs(directory, exist_ok=True)

    if type.lower() == 'png':
        mol_to_png(mol, img_file, size=(width,height))
    elif type.lower() == 'svg':
        mol_to_svg(mol, img_file, size=(width,height))
    return '<img src="{0}/{1}" style="width:{2}px;">'.format(directory, name, width)


def mol_to_pil(mol, size=(400, 200)):
    """
    Returns a Python Image Library (PIL) object containing an image of the given molecule's structure.
    """
    pil = MolToImage(mol, size=(size[0], size[1]))
    return pil


def mol_to_png(mol, name, size=(400, 200)):
    """
    Draws the molecule mol into a PNG file with filename 'name' and with the given size
    in pixels.
    """
    pil = mol_to_pil(mol, size)
    pil.save(name, 'PNG')


def mol_to_svg(mol, img_file, size=(400,200)):
    """
    Draw molecule mol's structure into an SVG file with path 'img_file' and with the
    given size.
    """
    img_wd, img_ht = size
    AllChem.Compute2DCoords(mol)
    try:
        mol.GetAtomWithIdx(0).GetExplicitValence()
    except RuntimeError:
        mol.UpdatePropertyCache(False)
    try:
        mc_mol = rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=True)
    except ValueError:
        # can happen on a kekulization failure
        mc_mol = rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=False)
    drawer = rdMolDraw2D.MolDraw2DSVG(img_wd, img_ht)
    drawer.DrawMolecule(mc_mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    svg = svg.replace('svg:','')
    svg = svg.replace('xmlns:svg','xmlns')
    with open(img_file, 'w') as img_out:
        img_out.write(svg)


def show_df(df):
    """
    Convenience function to display a pandas DataFrame in the current notebook window
    with HTML images rendered in table cells.
    """
    return HTML(df.to_html(escape=False))


def show_html(html):
    """
    Convenience function to display an HTML image specified by image tag 'html'.
    """
    return HTML(html)
