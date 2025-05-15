import atomsci.ddm.pipeline.predict_from_model as pfm
from atomsci.ddm.pipeline import model_pipeline as mp
from atomsci.ddm.pipeline import parameter_parser as parse
import pandas as pd
import numpy as np
import sklearn.metrics as skm

"""
make sure that the various ways of making predictions return
predictions in the same order as the input
"""
def test_predict_from_model():
    """test that predict_from_model makes predictions in the same
    order as the input
    """
    model_path = '../../examples/BSEP/models/bsep_classif_scaffold_split.tar.gz'
    csv_path = '../../examples/BSEP/data/ChEMBL25_BSEP_curated_data.csv'

    id_col = 'compound_id'
    smiles_col = 'base_rdkit_smiles'
    response_col = 'active'

    df = pd.read_csv(csv_path, dtype={id_col:str})
    #df = pd.concat([df.head(25),df.head(25)])
    #df = df.head(50)
    shuffled_df = df.sample(frac=1)

    pred_df = pfm.predict_from_model_file(model_path, shuffled_df, id_col=id_col, 
        smiles_col=smiles_col, response_col=response_col)

    old_id_col = shuffled_df[id_col].values
    new_id_col = pred_df[id_col].values

    match_rows = all([n == o for n, o in zip(new_id_col, old_id_col)])
    print('do all rows match?', match_rows)
    assert all([n == o for n, o in zip(new_id_col, old_id_col)])

    score = skm.accuracy_score(shuffled_df[response_col].values, pred_df[response_col+'_pred'].values)
    print('accuracy score', score)
    assert score > 0.5


def test_predict_full_dataset():
    """test that predict_full_dataset makes predictions in the same
    order as the input
    """
    model_path = '../../examples/BSEP/models/bsep_classif_scaffold_split.tar.gz'
    csv_path = '../../examples/BSEP/data/ChEMBL25_BSEP_curated_data.csv'

    id_col = 'compound_id'
    smiles_col = 'base_rdkit_smiles'
    response_col = 'active'
    conc_col = None
    is_featurized = False
    dont_standardize = False
    AD_method=None
    k=5
    dist_metric="euclidean"
    verbose = False

    df = pd.read_csv(csv_path, dtype={id_col:str})
    shuffled_df = df.sample(frac=1)

    input_df, pred_params = pfm._prepare_input_data(shuffled_df, id_col, smiles_col, response_col, 
        conc_col, dont_standardize, verbose)

    has_responses = ('response_cols' in pred_params)
    pred_params = parse.wrapper(pred_params)

    pipe = mp.create_prediction_pipeline_from_file(pred_params, reload_dir=None, model_path=model_path)
    pred_df = pipe.predict_full_dataset(input_df, contains_responses=has_responses, is_featurized=is_featurized,
                                        AD_method=AD_method, k=k, dist_metric=dist_metric)

    old_id_col = shuffled_df[id_col].values
    new_id_col = pred_df[id_col].values

    match_rows = np.all(new_id_col == old_id_col)
    print('do all rows match? ', match_rows)
    assert match_rows

    score = skm.accuracy_score(shuffled_df[response_col].values, pred_df[response_col+'_pred'].values)
    print('accuracy score ', score)
    assert score > 0.5

if __name__ == '__main__':
    test_predict_full_dataset()
    test_predict_from_model()
