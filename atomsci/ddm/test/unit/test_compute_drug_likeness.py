import pytest
import pandas as pd
from rdkit import Chem
from atomsci.ddm.utils.rdkit_easy import compute_drug_likeness

def test_compute_drug_likeness():
    # Create a DataFrame with sample SMILES strings
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

    # Compute drug likeness
    result_df = compute_drug_likeness(df, molecule_column='mol')

    # Check if the expected columns are present in the result DataFrame
    expected_columns = [
        'MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'TPSA', 'NumRotatableBonds',
        'MolarRefractivity', 'QED', 'TotalAtoms', 'Lipinski', 'Ghose', 'Veber'
    ]
    for col in expected_columns:
        assert col in result_df.columns

    # Check if the values are computed correctly for a known molecule (Ethanol)
    ethanol_row = result_df[result_df['smiles'] == 'CCO'].iloc[0]
    assert pytest.approx(ethanol_row['MolWt'], 0.1) == 46.07
    assert pytest.approx(ethanol_row['LogP'], 0.1) == -0.0014
    assert ethanol_row['NumHDonors'] == 1
    assert ethanol_row['NumHAcceptors'] == 1
    assert pytest.approx(ethanol_row['TPSA'], 0.1) == 20.23
    assert ethanol_row['NumRotatableBonds'] == 0
    assert pytest.approx(ethanol_row['MolarRefractivity'], 0.1) == 12.76
    assert pytest.approx(ethanol_row['QED'], 0.1) == 0.41
    assert ethanol_row['TotalAtoms'] == 9
    assert ethanol_row['Lipinski'] == True
    assert ethanol_row['Ghose'] == False
    assert ethanol_row['Veber'] == True