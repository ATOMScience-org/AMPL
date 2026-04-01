# smote/undersampling multitask test 

import os
import atomsci.ddm.pipeline.model_pipeline as mp
import atomsci.ddm.pipeline.parameter_parser as parse
import pytest


def test_mtss_model():
    script_path = os.path.dirname(os.path.realpath(__file__))
    dataset_file = os.path.join(script_path, "nanobret_multitask_classification_data.csv")
    split_uuid="e34ba827-a532-4313-9e63-8a9b0ed18ba9"
    odir = os.path.join(script_path, "output")

    id_col="compound_id"
    smiles_col="base_rdkit_smiles"
    response_cols="NEK1_active,NEK2_active,NEK3_active,NEK5_active,NEK9_active"

    params = {
        # logistics input
        "dataset_key": dataset_file, 
        "smiles_col": smiles_col,
        "prediction_type": "classification",
        "split_uuid": split_uuid, 
        "splitter": "multitaskscaffold",
        "response_cols": response_cols, 
        "previously_split": "True",

        # dataset
        "id_col": id_col,
        "result_dir": odir, 

        # featurization and model 
        "featurizer": "computed_descriptors",
        "descriptor_type": "rdkit_raw",
        "model_type": "NN",
        # grid search 
        "max_epochs": "300",
        "early_stopping_patience": "100",
        "sampling_method":"SMOTE",
        "layer_sizes": "128,128,128",
        "dropouts": "0.1,0.1,0.10",
        "learning_rates": "0.0007",

        # extras, can be deleted as needed
        "system": "LC", 
        "verbose": "True",
    }
    ampl_param = parse.wrapper(params)
    pl = mp.ModelPipeline(ampl_param)
    with pytest.raises(ValueError) as e:
        # this should say
        # Imbalanced-learn currently supports binary, multiclass and binarized encoded multiclasss targets. Multilabel and multioutput targets are not supported.
        pl.train_model()
    print("done")

def test_imblearn_mtss_compatibility():
    # this just shows that all SMOTE methods do not work with multitask problems.
    import sklearn.datasets as skdatasets
    import numpy as np
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE

    X, y = skdatasets.make_classification()
    print(X.shape, y.shape)
    multi_y = np.vstack([y, y, y]).transpose()
    print(multi_y.shape)

    for sampler in [SMOTE, ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE]:
        sm = sampler()
        try:
            _x, _y = sm.fit_resample(X, multi_y)
        except Exception as e:
            print(sm)
            print(e)

if __name__ == "__main__":
    test_mtss_model()
