import atomsci.ddm.pipeline.chem_diversity as cd
import pandas as pd

def test_check_ecfp_collisions():
    collision_smiles = [
        # group 1
        'CC(=O)N(C)[C@@H](Cc1ccccc1)C(=O)N(C)[C@@H](Cc1ccccc1)C(=O)N(C)[C@@H](Cc1ccccc1)C(N)=O',
        'CC(=O)N(C)[C@@H](Cc1ccccc1)C(=O)N(C)[C@@H](Cc1ccccc1)C(=O)N(C)[C@@H](Cc1ccccc1)C(=O)N(C)[C@@H](Cc1ccccc1)C(N)=O',
        'CC(=O)N(C)[C@@H](Cc1ccccc1)C(=O)N(C)[C@@H](Cc1ccccc1)C(N)=O',

        # group 2
        'CC(CN1CCCCCC1)NC(=O)/C=N/O',
        'CC(CN1CCCCC1)NC(=O)/C=N/O',

        # group 3
        'O=C(/C=N/O)NCCN1CCCCCC1',
        'O=C(/C=N/O)NCCN1CCCCC1',

        # group 4
        'COc1ccc(/C=C2\COc3cc(OCCCCCCCCNc4c5c(nc6ccccc46)CCCC5)ccc3C2=O)cc1',
        'COc1ccc(/C=C2\COc3cc(OCCCCCCNc4c5c(nc6ccccc46)CCCC5)ccc3C2=O)cc1'

        # group 5
        'O=c1ccc2ccc(OCCCCCN3CCN(CCCNc4c5c(nc6ccccc46)CCCC5)CC3)cc2o1',
        'O=c1ccc2ccc(OCCCCCCN3CCN(CCCNc4c5c(nc6ccccc46)CCCC5)CC3)cc2o1',

        # group 6
        'O/N=C/c1cc[n+](CCCCCC[n+]2ccc(/C=N/O)cc2)cc1',
        'O/N=C/c1cc[n+](CCCCC[n+]2ccc(/C=N/O)cc2)cc1',
        'O/N=C/c1cc[n+](CCCCCCC[n+]2ccc(/C=N/O)cc2)cc1',
    ]
    collision_df = pd.DataFrame(data={'smiles':collision_smiles})

    no_collision_smiles = [
        # group 1
        'CC(=O)N(C)[C@@H](Cc1ccccc1)C(=O)N(C)[C@@H](Cc1ccccc1)C(=O)N(C)[C@@H](Cc1ccccc1)C(N)=O',

        # group 2
        'CC(CN1CCCCCC1)NC(=O)/C=N/O',

        # group 3
        'O=C(/C=N/O)NCCN1CCCCCC1',

        # group 4
        'COc1ccc(/C=C2\COc3cc(OCCCCCCCCNc4c5c(nc6ccccc46)CCCC5)ccc3C2=O)cc1',

        # group 5
        'O=c1ccc2ccc(OCCCCCN3CCN(CCCNc4c5c(nc6ccccc46)CCCC5)CC3)cc2o1',

        # group 6
        'O/N=C/c1cc[n+](CCCCCC[n+]2ccc(/C=N/O)cc2)cc1',
    ]
    no_collision_df = pd.DataFrame(data={'smiles':no_collision_smiles})

    collision = cd.check_ecfp_collisions(collision_df, 'smiles', size=1024, radius=2)
    assert collision, "Collisions should have been found."

    no_collision = cd.check_ecfp_collisions(no_collision_df, 'smiles', size=1024, radius=2)
    assert not no_collision, "No collisions should have been found."

if __name__ == '__main__':
    test_check_ecfp_collisions()