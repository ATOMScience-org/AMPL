import atomsci.ddm.pipeline.featurization as featurization
import atomsci.ddm.pipeline.parameter_parser as parser
import pandas as pd

def get_collision_df():
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
        'COc1ccc(/C=C2\\COc3cc(OCCCCCCCCNc4c5c(nc6ccccc46)CCCC5)ccc3C2=O)cc1',
        'COc1ccc(/C=C2\\COc3cc(OCCCCCCNc4c5c(nc6ccccc46)CCCC5)ccc3C2=O)cc1'

        # group 5
        'O=c1ccc2ccc(OCCCCCN3CCN(CCCNc4c5c(nc6ccccc46)CCCC5)CC3)cc2o1',
        'O=c1ccc2ccc(OCCCCCCN3CCN(CCCNc4c5c(nc6ccccc46)CCCC5)CC3)cc2o1',

        # group 6
        'O/N=C/c1cc[n+](CCCCCC[n+]2ccc(/C=N/O)cc2)cc1',
        'O/N=C/c1cc[n+](CCCCC[n+]2ccc(/C=N/O)cc2)cc1',
        'O/N=C/c1cc[n+](CCCCCCC[n+]2ccc(/C=N/O)cc2)cc1',
    ]
    collision_df = pd.DataFrame(data={'smiles':collision_smiles, 'id':[f'id_{i}' for i in range(len(collision_smiles))]})

    return collision_df

def test_check_ecfp_collisions():
    collision_df = get_collision_df()

    no_collision_smiles = [
        # group 1
        'CC(=O)N(C)[C@@H](Cc1ccccc1)C(=O)N(C)[C@@H](Cc1ccccc1)C(=O)N(C)[C@@H](Cc1ccccc1)C(N)=O',

        # group 2
        'CC(CN1CCCCCC1)NC(=O)/C=N/O',

        # group 3
        'O=C(/C=N/O)NCCN1CCCCCC1',

        # group 4
        'COc1ccc(/C=C2\\COc3cc(OCCCCCCCCNc4c5c(nc6ccccc46)CCCC5)ccc3C2=O)cc1',

        # group 5
        'O=c1ccc2ccc(OCCCCCN3CCN(CCCNc4c5c(nc6ccccc46)CCCC5)CC3)cc2o1',

        # group 6
        'O/N=C/c1cc[n+](CCCCCC[n+]2ccc(/C=N/O)cc2)cc1',
    ]
    no_collision_df = pd.DataFrame(data={'smiles':no_collision_smiles})

    collision = featurization.check_ecfp_collisions(collision_df, 'smiles', size=1024, radius=2)
    assert collision, "Collisions should have been found."

    no_collision = featurization.check_ecfp_collisions(no_collision_df, 'smiles', size=1024, radius=2)
    assert not no_collision, "No collisions should have been found."

def test_dynamic_featurization_with_collisions():
    test_csv = "test_collision_df.csv"
    result_dir = "test_ecfp_collision_result"
    collision_df = get_collision_df()

    params = {
        # dataset key
        "dataset_key": test_csv,

        # columns
        "id_col": "id",
        "smiles_col": "smiles",
        "response_cols": "this column doesn't exist",

        # featurizer
        "featurizer": "ecfp",

        # result dir
        "result_dir": result_dir

    }
    pparams = parser.wrapper(params)
    dynamic_featurizer = featurization.DynamicFeaturization(pparams)

    # this should log a warning
    _ = dynamic_featurizer.featurize_data(collision_df, pparams, contains_responses=False)


if __name__ == '__main__':
    test_dynamic_featurization_with_collisions()
    test_check_ecfp_collisions()