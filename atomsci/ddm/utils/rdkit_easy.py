"""Utilities for clustering and visualizing compound structures using RDKit."""
# Mostly written by Logan Van Ravenswaay, with additions and edits by Ben Madej and Kevin McLoughlin.

import os

from IPython.display import HTML, display
from base64 import b64encode
import io

import logging

import pandas as pd
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import PandasTools
from rdkit.Chem.Draw import MolToImage, rdMolDraw2D
from rdkit.ML.Cluster import Butina
from rdkit.ML.Descriptors import MoleculeDescriptors

logging.basicConfig(format='%(asctime)-15s %(message)s')

def setup_notebook():
    """Set up current notebook for displaying plots and Bokeh output using full width of window"""
    from bokeh.plotting import output_notebook

    get_ipython().run_line_magic('matplotlib', 'inline')
    # Bokeh option
    output_notebook()
    display(HTML("<style>.container { width:100% !important; }</style>"))

    pd.set_option('display.max_columns', None)
    pd.set_option('max_seq_items', None)

def add_mol_column(df, smiles_col, molecule_col='mol'):
    """Converts SMILES strings in a data frame to RDKit Mol objects and adds them as a new column in the data frame.

    Args:
        df (pd.DataFrame): Data frame to add column to.

        smiles_col (str): Column containing SMILES strings.

        molecule_col (str): Name of column to create to hold Mol objects.

    Returns:
        pd.DataFrame: Modified data frame.

    """
    PandasTools.AddMoleculeColumnToFrame(df, smiles_col, molecule_col, includeFingerprints=True)
    return df


def calculate_descriptors(df, molecule_column='mol'):
    """Uses RDKit to compute various descriptors for compounds specified by Mol objects in the given data frame.

    Args:
        df (pd.DataFrame): Data frame containing molecules.

        molecule_column (str): Name of column containing Mol objects for compounds.

    Returns:
        pd.DataFrame: Modified data frame with added columns for the descriptors.

    """

    descriptors = [x[0] for x in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptors)
    cds=[]
    for mol in df[molecule_column]:
        cd = calculator.CalcDescriptors(mol)
        cds.append(cd)
    df2=pd.DataFrame(cds, columns=descriptors)
    df=df.join(df2, lsuffix='', rsuffix='_rdk')
    return df

def cluster_dataframe(df, molecule_column='mol', cluster_column='cluster', cutoff=0.2):
    """Performs Butina clustering on compounds specified by Mol objects in a data frame.

    Modifies the input dataframe to add a column 'cluster_column' containing the cluster
    index for each molecule.

    From RDKit cookbook http://rdkit.org/docs_temp/Cookbook.html.

    Args:
        df (pd.DataFrame): Data frame containing compounds to cluster.

        molecule_column (str): Name of column containing rdkit Mol objects for compounds.

        cluster_column (str): Column that will be created to hold cluster indices.

        cutoff (float): Maximum Tanimoto distance parameter used by Butina algorithm to identify neighbors of each molecule.

    Returns:
        None. Input data frame will be modified in place.

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
    df=df.copy()

def cluster_fingerprints(fps, cutoff=0.2):
    """Performs Butina clustering on compounds specified by a list of fingerprint bit vectors.

    From RDKit cookbook http://rdkit.org/docs_temp/Cookbook.html.

    Args:
        fps (list of rdkit.ExplicitBitVect): List of fingerprint bit vectors.

        cutoff (float): Cutoff distance parameter used to seed clusters in Butina algorithm.

    Returns:
        tuple of tuple: Indices of fingerprints assigned to each cluster.

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


def mol_to_html(mol, highlight=None, name='', type='svg', directory='rdkit_svg', embed=False, width=400, height=200):
    """Creates an image displaying the given molecule's 2D structure, and generates an HTML
    tag for it. The image can be embedded directly into the HTML tag or saved to a file.

    Args:
        mol (rdkit.Chem.Mol): Object representing molecule.

        highlight (rdkit.Chem.Mol): Optional object representing a set of atoms and bonds to be highlighted in
        the image.

        name (str): Filename of image file to create, relative to 'directory'; only used if embed=False.

        type (str): Image format; must be 'png' or 'svg'.

        directory (str): Path relative to notebook directory of subdirectory where image file will be written.
        The directory will be created if necessary. Note that absolute paths will not work in notebooks. Ignored if embed=True.

        embed (bool): If True, image data will be embedded in the generated HTML tag. Otherwise it will be written to a
        file determined by the `directory` and `name` arguments.

        width (int): Width of image bounding box.

        height (int): Height of image bounding box.

    Returns:
        str: HTML image tag referencing the image file.

    """
    log = logging.getLogger('ATOM')
    
    if type.lower() not in ['png','svg']:
        log.warning('Image type not recognized. Choose between png and svg.')
        return ''
    
    # Handle Mol generated from invalid SMILES string
    if mol is None:
        return ''

    if embed:
        if type.lower() == 'png':
            img=mol_to_pil(mol, size=(width,height), highlight=highlight)
            imgByteArr = io.BytesIO()
            img.save(imgByteArr, format=img.format)
            imgByteArr = imgByteArr.getvalue()
            data_url = 'data:image/png;base64,' + b64encode(imgByteArr).decode()
        elif type.lower() == 'svg':
            img=mol_to_svg(mol, size=(width,height), highlight=highlight).encode('utf-8')
            data_url = 'data:image/svg+xml;base64,' + b64encode(img).decode()
        return f"<img src='{data_url}' style='width:{width}px;'>" 
    
    else:
        img_file = '%s/%s' % (directory, name)
        os.makedirs(directory, exist_ok=True)
        if type.lower() == 'png':
            save_png(mol, img_file, size=(width,height), highlight=highlight)
        elif type.lower()=='svg':
            save_svg(mol, img_file, size=(width,height), highlight=highlight)
        return f"<img src='{img_file}' style='width:{width}px;'>"

def matching_atoms_and_bonds(mol, match_mol):
    """Returns lists of indices of atoms and bonds within molecule `mol` that are part of the substructure
    matched by `match_mol`.

    Args:
        mol (rdkit.Chem.Mol): Object representing molecule.

        match_mol (rdkit.Chem.Mol): Object representing a substructure or SMARTS pattern to be
        compared against `mol`, typically created by `Chem.MolFromSmiles()` or `Chem.MolFromSmarts()`.

    Returns:
        match_atoms, match_bonds (tuple(list(int), list(int))): Lists of indices of atoms and bonds
        within `mol` contained in the substructure (if any) matched by `match_mol`. Returns empty lists
        if there is no match.
    """
    match_atoms = []
    match_bonds = []
    if match_mol is not None:
        match_atoms = list(mol.GetSubstructMatch(match_mol))
        if len(match_atoms) > 1:
            for bond in match_mol.GetBonds():
               aid1 = match_atoms[bond.GetBeginAtomIdx()]
               aid2 = match_atoms[bond.GetEndAtomIdx()]
               match_bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())
    return match_atoms, match_bonds


def mol_to_pil(mol, size=(400, 200), highlight=None):
    """Returns a Python Image Library (PIL) object containing an image of the given molecule's structure.

    Args:
        mol (rdkit.Chem.Mol): Object representing molecule.

        size (tuple): Width and height of bounding box of image.

        highlight (rdkit.Chem.Mol): Object representing substructure to highlight on molecule.

    Returns:
        PIL.PngImageFile: An object containing an image of the molecule's structure.

    """
    highlight_atoms, highlight_bonds = matching_atoms_and_bonds(mol, highlight)
    pil = MolToImage(mol, size=(size[0], size[1]), highlightAtoms=highlight_atoms, highlightBonds=highlight_bonds)
    return pil


def save_png(mol, name, size=(400, 200), highlight=None):
    """Draws the molecule mol into a PNG file with filename 'name' and with the given size
    in pixels.

    Args:
        mol (rdkit.Chem.Mol): Object representing molecule.

        name (str): Path to write PNG file to.

        size (tuple): Width and height of bounding box of image.

        highlight (rdkit.Chem.Mol): Object representing substructure to highlight on molecule.

    """
    pil = mol_to_pil(mol, size, highlight=highlight)
    pil.save(name, 'PNG')


def mol_to_svg(mol, size=(400,200), highlight=None):
    """Returns a RDKit MolDraw2DSVG object containing an image of the given molecule's structure.

    Args:
        mol (rdkit.Chem.Mol): Object representing molecule.

        size (tuple): Width and height of bounding box of image.

        highlight (rdkit.Chem.Mol): Object representing substructure to highlight on molecule.

    Returns:
       RDKit.rdMolDraw2D.MolDraw2DSVG text (str): An SVG object containing an image of the molecule's structure.

    """
    img_wd, img_ht = size
    highlight_atoms, highlight_bonds = matching_atoms_and_bonds(mol, highlight)
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
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mc_mol, highlightAtoms=highlight_atoms,
                                                    highlightBonds=highlight_bonds)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    svg = svg.replace('svg:','')
    svg = svg.replace('xmlns:svg','xmlns')
    return svg


def save_svg(mol, name, size=(400,200), highlight=None):
    """Draws the molecule mol into an SVG file with filename 'name' and with the given size
    in pixels.

    Args:
        mol (rdkit.Chem.Mol): Object representing molecule.

        name (str): Path to write SVG file to.

        size (tuple): Width and height of bounding box of image.

        highlight (rdkit.Chem.Mol): Object representing substructure to highlight on molecule.

    """
    svg = mol_to_svg(mol, size, highlight=highlight)
    with open(name, 'w') as img_out:
        img_out.write(svg)


def show_df(df):
    """Convenience function to display a pandas DataFrame in the current notebook window
    with HTML images rendered in table cells.

    Args:
        df (pd.DataFrame): Data frame to display.

    Returns:
        None

    """
    return HTML(df.to_html(escape=False))


def show_html(html):
    """Convenience function to display an HTML image specified by image tag 'html'.

    Args:
        html (str): HTML image tag to render.

    Returns:
        None

    """
    return HTML(html)

