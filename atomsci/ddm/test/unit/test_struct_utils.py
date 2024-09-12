import atomsci.ddm.utils.struct_utils as su
import os
import numpy as np

test_smiles = ['Brc1cc(O[C@@H]2C[C@H]3CC[C@@H](C2)N3)cc(-c2ccccc2)c1',
                'Brc1ccc(N2CCN(CCCCOc3ccc4ccccc4c3)CC2)cc1',
                'Brc1ccc(N2CCN(CCCN3CCC(Cc4ccccc4)CC3)CC2)cc1',
                ]


def test_get_rdkit_smiles():
    result = su.get_rdkit_smiles('asdfasdfasdf', useIsomericSmiles=True)
    assert result == ''

    result = su.get_rdkit_smiles(test_smiles[0], useIsomericSmiles=True)
    assert result == 'Brc1cc(O[C@@H]2C[C@H]3CC[C@@H](C2)N3)cc(-c2ccccc2)c1'

    result = su.get_rdkit_smiles(test_smiles[0], useIsomericSmiles=False)
    assert result == 'Brc1cc(OC2CC3CCC(C2)N3)cc(-c2ccccc2)c1'

def test_rdkit_smiles_from_smiles():
    result = su.rdkit_smiles_from_smiles(test_smiles, useCanonicalTautomers=True)
    assert result == ['Brc1cc(O[C@@H]2C[C@H]3CC[C@@H](C2)N3)cc(-c2ccccc2)c1', 
        'Brc1ccc(N2CCN(CCCCOc3ccc4ccccc4c3)CC2)cc1', 
        'Brc1ccc(N2CCN(CCCN3CCC(Cc4ccccc4)CC3)CC2)cc1']

    result = su.rdkit_smiles_from_smiles(test_smiles, useCanonicalTautomers=False)
    assert result == ['Brc1cc(O[C@@H]2C[C@H]3CC[C@@H](C2)N3)cc(-c2ccccc2)c1',
        'Brc1ccc(N2CCN(CCCCOc3ccc4ccccc4c3)CC2)cc1', 
        'Brc1ccc(N2CCN(CCCN3CCC(Cc4ccccc4)CC3)CC2)cc1']

    result = su.rdkit_smiles_from_smiles(test_smiles, useIsomericSmiles=False)
    assert result == ['Brc1cc(OC2CC3CCC(C2)N3)cc(-c2ccccc2)c1',
        'Brc1ccc(N2CCN(CCCCOc3ccc4ccccc4c3)CC2)cc1',
        'Brc1ccc(N2CCN(CCCN3CCC(Cc4ccccc4)CC3)CC2)cc1']

    result = su.rdkit_smiles_from_smiles(test_smiles, useIsomericSmiles=True)
    assert result == ['Brc1cc(O[C@@H]2C[C@H]3CC[C@@H](C2)N3)cc(-c2ccccc2)c1',
        'Brc1ccc(N2CCN(CCCCOc3ccc4ccccc4c3)CC2)cc1',
        'Brc1ccc(N2CCN(CCCN3CCC(Cc4ccccc4)CC3)CC2)cc1']

    result = su.rdkit_smiles_from_smiles(test_smiles, workers=2)
    assert result == ['Brc1cc(O[C@@H]2C[C@H]3CC[C@@H](C2)N3)cc(-c2ccccc2)c1',
        'Brc1ccc(N2CCN(CCCCOc3ccc4ccccc4c3)CC2)cc1',
        'Brc1ccc(N2CCN(CCCN3CCC(Cc4ccccc4)CC3)CC2)cc1']

    result = su.rdkit_smiles_from_smiles(['asdfasdf']+test_smiles, workers=2)
    assert result == ['',
        'Brc1cc(O[C@@H]2C[C@H]3CC[C@@H](C2)N3)cc(-c2ccccc2)c1',
        'Brc1ccc(N2CCN(CCCCOc3ccc4ccccc4c3)CC2)cc1',
        'Brc1ccc(N2CCN(CCCN3CCC(Cc4ccccc4)CC3)CC2)cc1']

def test_mols_from_smiles():
    _ = su.mols_from_smiles(test_smiles)

    _ = su.mols_from_smiles(test_smiles, workers=2)

def test_base_smiles_from_smiles():
    results = su.base_smiles_from_smiles(test_smiles, workers=2)
    assert results == ['Brc1cc(O[C@@H]2C[C@H]3CC[C@@H](C2)N3)cc(-c2ccccc2)c1',
        'Brc1ccc(N2CCN(CCCCOc3ccc4ccccc4c3)CC2)cc1',
        'Brc1ccc(N2CCN(CCCN3CCC(Cc4ccccc4)CC3)CC2)cc1']

    results = su.base_smiles_from_smiles(test_smiles,
        useCanonicalTautomers=True,
        useIsomericSmiles=False,
        removeCharges=True,
        workers=2)
    assert results == ['Brc1cc(OC2CC3CCC(C2)N3)cc(-c2ccccc2)c1',
        'Brc1ccc(N2CCN(CCCCOc3ccc4ccccc4c3)CC2)cc1',
        'Brc1ccc(N2CCN(CCCN3CCC(Cc4ccccc4)CC3)CC2)cc1']

    results = su.base_smiles_from_smiles(['asdf']+test_smiles,
        useCanonicalTautomers=True,
        useIsomericSmiles=False,
        removeCharges=True,
        workers=2)
    assert results == ['',
        'Brc1cc(OC2CC3CCC(C2)N3)cc(-c2ccccc2)c1',
        'Brc1ccc(N2CCN(CCCCOc3ccc4ccccc4c3)CC2)cc1',
        'Brc1ccc(N2CCN(CCCN3CCC(Cc4ccccc4)CC3)CC2)cc1']

def test_kekulize_smiles():
    results = su.kekulize_smiles(test_smiles)
    assert results==['BrC1=CC(O[C@@H]2C[C@H]3CC[C@@H](C2)N3)=CC(C2=CC=CC=C2)=C1',
        'BrC1=CC=C(N2CCN(CCCCOC3=CC=C4C=CC=CC4=C3)CC2)C=C1',
        'BrC1=CC=C(N2CCN(CCCN3CCC(CC4=CC=CC=C4)CC3)CC2)C=C1']

    results = su.kekulize_smiles(['asdf']+test_smiles, useIsomericSmiles=False)
    assert results==['',
        'BrC1=CC(OC2CC3CCC(C2)N3)=CC(C2=CC=CC=C2)=C1',
        'BrC1=CC=C(N2CCN(CCCCOC3=CC=C4C=CC=CC4=C3)CC2)C=C1',
        'BrC1=CC=C(N2CCN(CCCN3CCC(CC4=CC=CC=C4)CC3)CC2)C=C1']

def test_base_mol_from_smiles():
    _ = su.base_mol_from_smiles(test_smiles[0])

    _ = su.base_mol_from_smiles(test_smiles[0], removeCharges=True)

    result = su.base_mol_from_smiles(['asdf'], removeCharges=True)
    assert result is None

    result = su.base_mol_from_smiles(True)
    assert result is None

    result = su.base_mol_from_smiles(True)
    assert result is None

    result = su.base_mol_from_smiles('')
    assert result is None

def test_base_mol_smiles_from_inchi():
    test_inchis = ['InChI=1S/C6H8O6/c7-1-2(8)5-3(9)4(10)6(11)12-5/h2,5,7-10H,1H2/t2-,5+/m0/s1',
        'InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3']

    _ = su.base_mol_from_inchi(test_inchis)
    _ = su.base_mol_from_inchi(test_inchis, useIsomericSmiles=False, removeCharges=True)

    results = su.base_smiles_from_inchi(test_inchis)
    assert results==['O=C1O[C@H]([C@@H](O)CO)C(O)=C1O', 'CCO']

    results = su.base_smiles_from_inchi(test_inchis,
                useIsomericSmiles=False,
                removeCharges=True)
    assert results==['O=C1OC(C(O)CO)C(O)=C1O', 'CCO']

    result = su.base_mol_from_inchi(True)
    assert result is None

    result = su.base_mol_from_inchi('')
    assert result is None

def test_draw_structure():
    test_png_path = 'draw_structure.png'
    assert not os.path.exists(test_png_path)
    su.draw_structure(test_smiles[0], test_png_path)
    assert os.path.exists(test_png_path)
    os.remove(test_png_path)

    assert not os.path.exists(test_png_path)
    su.draw_structure('asdfsdf', test_png_path)
    assert not os.path.exists(test_png_path)

def test_smiles_to_inchi_key():
    test_inchis = [su.smiles_to_inchi_key(s) for s in ['asdf']+test_smiles]
    assert test_inchis==[None,
        'OKLKBYAVPPRKEG-IZZQQSIFSA-N',
        'PHWGAEVYRCQNEM-UHFFFAOYSA-N',
        'PVGPGTOPBMMJHX-UHFFFAOYSA-N']

def test_mol_wt_from_smiles():
    mol_wts = [su.mol_wt_from_smiles(s) for s in ['asdf']+test_smiles]

    assert np.isnan(mol_wts[0])

    assert mol_wts[1:]==[358.27900000000005, 439.3970000000001, 456.47200000000026]

def test_canonical_tautomers_from_smiles():
    canonical_tautomers = [su.canonical_tautomers_from_smiles(s) for s in ['asdf']+test_smiles]
    assert canonical_tautomers==[[''], 
        ['Brc1cc(O[C@@H]2C[C@H]3CC[C@@H](C2)N3)cc(-c2ccccc2)c1'],
        ['Brc1ccc(N2CCN(CCCCOc3ccc4ccccc4c3)CC2)cc1'],
        ['Brc1ccc(N2CCN(CCCN3CCC(Cc4ccccc4)CC3)CC2)cc1']]

    canonical_tautomer = su.canonical_tautomers_from_smiles(test_smiles[0])
    assert canonical_tautomer==['Brc1cc(O[C@@H]2C[C@H]3CC[C@@H](C2)N3)cc(-c2ccccc2)c1']

if __name__ == '__main__':

    test_get_rdkit_smiles()
    test_rdkit_smiles_from_smiles()
    test_mols_from_smiles()
    test_base_smiles_from_smiles()
    test_kekulize_smiles()
    test_base_mol_from_smiles()
    test_base_mol_smiles_from_inchi()
    test_draw_structure()
    test_smiles_to_inchi_key()
    test_mol_wt_from_smiles()
    test_canonical_tautomers_from_smiles()