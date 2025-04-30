import pandas as pd
from rdkit import Chem
from atomsci.ddm.utils.rdkit_easy import calculate_descriptors, compute_drug_likeness, add_mol_column, cluster_dataframe, cluster_fingerprints
from rdkit.Chem import AllChem
import pytest

@pytest.fixture
def sample_data():
    """
    Fixture that provides sample data for testing.
    Returns:
        pd.DataFrame: A DataFrame containing sample SMILES strings and their corresponding RDKit Mol objects.
    """
    data = {
        'smiles': [
            'CCO',  # Ethanol
            'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
            'C1=CC=C(C=C1)C=O',  # Benzaldehyde
            'CC(C)NCC(O)COC1=CC=CC=C1'  # Pseudoephedrine
        ]
    }
    df = pd.DataFrame(data)
    df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)
    return df

def test_calculate_descriptors(sample_data):
    """
    Test the `calculate_descriptors` function to ensure it calculates the correct molecular descriptors.
    Args:
        sample_data (pd.DataFrame): The sample data fixture containing SMILES strings and RDKit Mol objects.
    Asserts:
        - The result DataFrame contains the expected descriptor columns.
        - The descriptor values for Ethanol are computed correctly.
    """
    result_df = calculate_descriptors(sample_data, molecule_column='mol')
    # inspect result_df columns
    expected_columns = [
         'MolWt', 'qed', 'MaxAbsEStateIndex', 'MaxEStateIndex', 'HeavyAtomMolWt', 'SPS'
    ]
    for col in expected_columns:
        assert col in result_df.columns

    # Check if the values are computed correctly for a known molecule (Ethanol)
    ethanol_row = result_df[result_df['smiles'] == 'CCO'].iloc[0]
    assert pytest.approx(ethanol_row['MolWt'], 0.1) == 46.07
    assert ethanol_row['NumHDonors'] == 1
    assert ethanol_row['NumHAcceptors'] == 1
    assert pytest.approx(ethanol_row['TPSA'], 0.1) == 20.23
    assert ethanol_row['NumRotatableBonds'] == 0
    assert pytest.approx(ethanol_row['qed'], 0.1) == 0.41
    
def test_compute_drug_likeness(sample_data):
    """
    Test the `compute_drug_likeness` function to ensure it calculates the correct drug-likeness properties.
    Args:
        sample_data (pd.DataFrame): The sample data fixture containing SMILES strings and RDKit Mol objects.
    Asserts:
        - The result DataFrame contains the expected drug-likeness property columns.
        - The drug-likeness properties for Ethanol are computed correctly.
    """
    result_df = compute_drug_likeness(sample_data, molecule_column='mol')
    # inspect result_df columns
    expected_columns = [
        'MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'TPSA', 'NumRotatableBonds',
        'MolarRefractivity', 'QED', 'TotalAtoms', 'Lipinski', 'Ghose', 'Veber'
    ]
    for col in expected_columns:
        assert col in result_df.columns

    # Check if the values are computed correctly for a known molecule (Ethanol)
    ethanol_row = result_df[result_df['smiles'] == 'CCO'].iloc[0]
    assert pytest.approx(ethanol_row['MolWt'], 0.1) == 46.07
    assert pytest.approx(ethanol_row['LogP'], abs=0.01) == -0.001
    assert ethanol_row['NumHDonors'] == 1
    assert ethanol_row['NumHAcceptors'] == 1
    assert pytest.approx(ethanol_row['TPSA'], 0.1) == 20.23
    assert ethanol_row['NumRotatableBonds'] == 0
    assert pytest.approx(ethanol_row['MolarRefractivity'], 0.1) == 12.32
    assert pytest.approx(ethanol_row['QED'], 0.1) == 0.41
    assert ethanol_row['TotalAtoms'] == 9
    assert bool(ethanol_row['Lipinski']) is True
    assert bool(ethanol_row['Ghose']) is False
    assert bool(ethanol_row['Veber']) is True
    
def test_cluster_dataframe(sample_data):
    """
    Test the `cluster_dataframe` function to ensure it correctly clusters molecules in a DataFrame.
    Args:
        sample_data (pd.DataFrame): The sample data fixture containing SMILES strings and RDKit Mol objects.
    Asserts:
        - The 'cluster' column is added to the DataFrame.
        - All molecules are assigned to a cluster.
        - There is more than one cluster.
        - Molecules with different structures are in different clusters.
    """
    # Add a column with RDKit Mol objects
    sample_data = add_mol_column(sample_data, 'smiles', 'mol')
        
    # Perform clustering
    cluster_dataframe(sample_data, molecule_column='mol', cluster_column='cluster', cutoff=0.2)
        
    # Check if the 'cluster' column is added
    assert 'cluster' in sample_data.columns
        
    # Check if all molecules are assigned to a cluster
    assert sample_data['cluster'].isnull().sum() == 0
        
    # Check if the clusters are consistent
    clusters = sample_data['cluster'].unique()
    assert len(clusters) > 1  # Ensure there is more than one cluster
        
    # Check if molecules with similar structures are in the same cluster
    ethanol_cluster = sample_data[sample_data['smiles'] == 'CCO']['cluster'].iloc[0]
    benzaldehyde_cluster = sample_data[sample_data['smiles'] == 'C1=CC=C(C=C1)C=O']['cluster'].iloc[0]
    assert ethanol_cluster != benzaldehyde_cluster  # Ethanol and Benzaldehyde should be in different clusters
    
def test_cluster_fingerprints():
    """
    Test the `cluster_fingerprints` function to ensure it correctly clusters molecular fingerprints.
    Asserts:
        - There is more than one cluster.
        - Molecules with different structures are in different clusters.
        - All molecules are clustered.
    """
    # Create sample fingerprints
    mols = [
        Chem.MolFromSmiles('CCO'),  # Ethanol
        Chem.MolFromSmiles('CC(=O)OC1=CC=CC=C1C(=O)O'),  # Aspirin
        Chem.MolFromSmiles('CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'),  # Ibuprofen
        Chem.MolFromSmiles('C1=CC=C(C=C1)C=O'),  # Benzaldehyde
        Chem.MolFromSmiles('CC(C)NCC(O)COC1=CC=CC=C1')  # Pseudoephedrine
    ]
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024) for mol in mols]

    # Perform clustering
    clusters = cluster_fingerprints(fps, cutoff=0.2)

    # Check if the clusters are consistent
    assert len(clusters) > 1  # Ensure there is more than one cluster

    # Check if molecules with similar structures are in the same cluster
    ethanol_cluster = [i for i, cluster in enumerate(clusters) if 0 in cluster][0]
    benzaldehyde_cluster = [i for i, cluster in enumerate(clusters) if 3 in cluster][0]
    assert ethanol_cluster != benzaldehyde_cluster  # Ethanol and Benzaldehyde should be in different clusters

    # Check if the clusters contain the correct number of molecules
    total_molecules = sum(len(cluster) for cluster in clusters)
    assert total_molecules == len(mols)  # Ensure all molecules are clustered