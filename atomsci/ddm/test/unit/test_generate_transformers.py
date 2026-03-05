import atomsci.ddm.utils.generate_transformers as gt

def test_filter_outlier_features():
    dataset_key = '../test_datasets/Molport_test.csv'
    id_col = 'compound_id'
    smiles_col = 'base_rdkit_smiles'
    response_cols = 'dummy_response'

    a, b, c = gt.filter_outlier_features(
        dataset_key, id_col, smiles_col, response_cols, 
        featurizer='computed_descriptors', descriptor_type='rdkit_raw', 
        threshold=1e10)

    expected_a = ['MolPort-008-351-280', 'MolPort-008-351-290', 'MolPort-008-351-328',
                  'MolPort-008-351-372', 'MolPort-008-351-275', 'MolPort-008-351-375',
                  'MolPort-008-351-277', 'MolPort-008-351-379', 'MolPort-008-351-322',
                  'MolPort-008-351-297', 'MolPort-008-351-353', 'MolPort-008-351-336',
                  'MolPort-008-351-294', 'MolPort-008-351-363', 'MolPort-008-351-321',
                  'MolPort-008-351-357', 'MolPort-008-351-318', 'MolPort-008-351-354',
                  'MolPort-008-351-295', 'MolPort-008-351-296', 'MolPort-008-351-338',
                  'MolPort-008-351-382', 'MolPort-008-351-339', 'MolPort-008-351-292',
                  'MolPort-008-351-313', 'MolPort-008-351-371', 'MolPort-008-351-359',
                  'MolPort-008-351-283', 'MolPort-008-351-344', 'MolPort-008-351-343',
                  'MolPort-008-351-305', 'MolPort-008-351-352', 'MolPort-008-351-319',
                  'MolPort-008-351-317', 'MolPort-008-351-358', 'MolPort-008-351-373']
    expected_b = ['Ipc']*len(expected_a)

    assert list(a) == expected_a
    assert list(b) == expected_b
    assert all([val > 1e10 for val in c])

def test_filter_outlier_MW():
    dataset_key = '../test_datasets/Molport_test.csv'
    smiles_col = 'base_rdkit_smiles'

    outliers = gt.filter_outlier_MW(dataset_key, smiles_col, threshold=450, workers=1)

    expected_outliers = ['O=C(CC[C@@H]1NC(=O)N(Cc2ccc3c(c2)OCO3)C1=O)NCCC(c1ccccc1)c1ccccc1', 
                         'Cc1[nH]c(=O)[nH]c(=O)c1S(=O)(=O)N1CCCC(C(=O)Nc2ccc(C(F)(F)F)cc2)C1', 
                         'O=C1CC2(C(=O)Nc3ccc(Br)cc32)c2sc(C(=O)O)c(-c3ccccc3)c2N1', 
                         'O=C(CC[C@@H]1NC(=O)N(Cc2ccc3c(c2)OCO3)C1=O)NC1CCN(Cc2ccccc2)CC1', 
                         'COC(=O)C[C@H](NC(=O)N1CCc2nc[nH]c2C1c1cccc(Cl)c1Cl)C(=O)OC', 
                         'Cc1[nH]c(=O)[nH]c(=O)c1S(=O)(=O)N1CCCC(C(=O)NCCc2ccc(Cl)cc2Cl)C1', 
                         'CN1CCN(C2(CNC(=O)C[C@@H]3NC(=O)N(Cc4ccc5c(c4)OCO5)C3=O)CCCCC2)CC1', 
                         'O=C(C[C@@H]1NC(=O)N(Cc2ccc3c(c2)OCO3)C1=O)NCc1ccc(Cl)cc1Cl']

    assert outliers == expected_outliers

    print(outliers)

if __name__ == "__main__":
    test_filter_outlier_features()
    test_filter_outlier_MW()