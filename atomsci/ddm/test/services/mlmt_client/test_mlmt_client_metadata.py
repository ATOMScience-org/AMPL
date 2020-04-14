from utilities import *

# Run with `python3 -m pytest -s` in the directory this file is in.
# Run a specific test with `python3 -m pytest -s test_mlmt_client_metadata.py
# -k <test_name>`
# Also test using Swagger UI or by calling these functions from an IPython
# notebook.


# https://docs.pytest.org/en/latest/getting-started.html for more on test
# classes.
class TestMetadataSuccess(object):
    # https://docs.pytest.org/en/latest/xunit_setup.html for more on setup
    # methods.
    def setup_method(self):
        setup(self)

    def teardown_method(self):
        teardown(self)

    def test_metadata_success_regular_syntax_match(self):
        d = save_dicts_1(self.client_wrapper)
        # Test regular syntax.
        filter_dict = {'model_uuid': 'uuid_1'}
        check(self.client_wrapper, filter_dict, 'get', 'metadata', 200,
              match_list=[d])

    def test_metadata_success_regular_syntax_non_match(self):
        d = save_dicts_1(self.client_wrapper)
        # Test regular syntax.
        filter_dict = {'model_uuid': 'uuid_2'}
        check(self.client_wrapper, filter_dict, 'get', 'metadata', 200,
              non_match_list=[d])

    def test_metadata_success_mongo_syntax(self):
        d = save_dicts_1(self.client_wrapper)
        # Test mongo syntax.
        filter_dict = {
            'model_uuid' : 'uuid_1',
            'ModelMetadata.TrainingDataset.dataset_key': {'$eq': 2},
            'ModelMetadata.TrainingDataset.dataset_bucket': {'$gt': 2},
            'ModelMetadata.TrainingDataset.dataset_oid': {'$gte': 2},
            'ModelMetadata.TrainingDataset.class_names': {'$in': [1, 2, 3]},
            'ModelMetadata.TrainingDataset.num_classes': {'$lt': 2},
            'ModelMetadata.TrainingDataset.feature_transform_type': {'$lte': 2},
            'ModelMetadata.TrainingDataset.response_transform_type': {'$ne': 3},
            'ModelMetadata.TrainingDataset.id_col' : {'$nin' : [0, 1, 3, 4]}
        }
        check(self.client_wrapper, filter_dict, 'get', 'metadata', 200,
              match_list=[d])

    def test_metadata_success_primary_syntax(self):
        d = save_dicts_1(self.client_wrapper)
        # Test primary syntax.
        filter_dict = {
            'model_uuid' : 'uuid_1',
            'ModelMetadata.TrainingDataset.dataset_key': ['=', 2],
            'ModelMetadata.TrainingDataset.dataset_bucket': ['>', 2],
            'ModelMetadata.TrainingDataset.dataset_oid': ['>=', 2],
            'ModelMetadata.TrainingDataset.class_names': ['in', [1, 2, 3]],
            'ModelMetadata.TrainingDataset.num_classes': ['<', 2],
            'ModelMetadata.TrainingDataset.feature_transform_type': ['<=', 2],
            'ModelMetadata.TrainingDataset.response_transform_type': ['!=', 3],
            'ModelMetadata.TrainingDataset.id_col' : ['nin', [0, 1, 3, 4]]
        }
        check(self.client_wrapper, filter_dict, 'get', 'metadata', 200,
              match_list=[d])

    def test_metadata_success_non_existent_key(self):
        d = save_dicts_1(self.client_wrapper)
        # Test non-existent key.
        filter_dict = {'ModelMetadata.TrainingDataset.not_a_key' : 5}
        check(self.client_wrapper, filter_dict, 'get', 'metadata', 200,
              non_match_list=[d])

    def test_metadata_success_no_filters(self):
        (d_1, d_2, d_3, d_4) = save_dicts_2(self.client_wrapper)
        # Test no filters.
        check(self.client_wrapper, {}, 'get', 'metadata', 200,
              match_list=[d_1, d_2, d_3, d_4])

    def test_metadata_success_min_one_key(self):
        (d_1, d_2, d_3, d_4) = save_dicts_2(self.client_wrapper)
        # Test minimum on one key.
        filter_dict = \
            {'ModelMetadata.TrainingDataset.dataset_key': ['min', None]}
        check(self.client_wrapper, filter_dict, 'get', 'metadata', 200,
              match_list=[d_1], non_match_list=[d_2, d_3, d_4])

    def test_metadata_success_max_one_key(self):
        (d_1, d_2, d_3, d_4) = save_dicts_2(self.client_wrapper)
        # Test maximum on one key.
        filter_dict = {'ModelMetadata.TrainingDataset.dataset_key' : ['max', None]}
        check(self.client_wrapper, filter_dict, 'get', 'metadata', 200,
              match_list=[d_4], non_match_list=[d_1, d_2, d_3])

    def test_metadata_success_extrema_diff_keys_same_doc(self):
        (d_1, d_2, d_3, d_4) = save_dicts_2(self.client_wrapper)
        # Test minimum and maximum on different keys -
        # min and max occur on same metadata document.
        filter_dict = {
            'ModelMetadata.TrainingDataset.dataset_key': ['min', 1],
            'ModelMetadata.TrainingDataset.id_col': ['max', 2]
        }
        check(self.client_wrapper, filter_dict, 'get', 'metadata', 200,
              match_list=[d_1], non_match_list=[d_2, d_3, d_4])

    def test_metadata_success_extrema_diff_keys_diff_docs(self):
        (d_1, d_2, d_3, d_4) = save_dicts_2(self.client_wrapper)
        # Test minimum and maximum on different keys -
        # min and max occur on different metadata documents.
        filter_dict = {
            'ModelMetadata.TrainingDataset.dataset_key':
                ['min', 1],  # d_1
            'ModelMetadata.TrainingDataset.feature_transform_type':
                ['max', 2]  # d_3
        }
        check(self.client_wrapper, filter_dict, 'get', 'metadata', 200,
              match_list=[d_1], non_match_list=[d_2, d_3, d_4])

    def test_metadata_success_max_two_matches(self):
        (d_1, d_2, d_3, d_4) = save_dicts_2(self.client_wrapper)
        # Test returning two dicts for maximum.
        filter_dict = {'ModelMetadata.TrainingDataset.response_transform_type':
                       ['max', None]}
        check(self.client_wrapper, filter_dict, 'get', 'metadata', 200,
              match_list=[d_3, d_4], non_match_list=[d_1, d_2])

    def test_metadata_success_min_extra_filter(self):
        (d_1, d_2, d_3, d_4) = save_dicts_2(self.client_wrapper)
        # Test max with extra filter
        filter_dict = {
            'ModelMetadata.TrainingDataset.dataset_key': ['min', None],
            'ModelMetadata.TrainingDataset.num_classes': 1
        }
        check(self.client_wrapper, filter_dict, 'get', 'metadata', 200,
              match_list=[d_1], non_match_list=[d_2, d_3, d_4])

    def test_metadata_success_min_diff_keys(self):
        (d_1, d_2, d_3, d_4) = save_dicts_2(self.client_wrapper)
        # Test minimum on two different keys.
        filter_dict = {
            'ModelMetadata.TrainingDataset.dataset_key': ['min', 1],
            'ModelMetadata.TrainingDataset.dataset_bucket': ['min', 2]
        }
        check(self.client_wrapper, filter_dict, 'get', 'metadata', 200,
              match_list=[d_1], non_match_list=[d_2, d_3, d_4])

    def test_metadata_success_min_non_existent_key(self):
        (d_1, d_2, d_3, d_4) = save_dicts_2(self.client_wrapper)
        # Test minimum on non_existent key
        filter_dict = {'ModelMetadata.TrainingDataset.not_a_key':
                           ['min', None]}
        check(self.client_wrapper, filter_dict, 'get', 'metadata', 200,
              match_list=[], non_match_list=[d_1, d_2, d_3, d_4])

    def test_metadata_success_min_subset(self):
        (d_1, d_2, d_3, d_4) = save_dicts_2(self.client_wrapper)
        # Test minimum for a subset.
        filter_dict = {
            # d_1 matches, but d_3 matches with the following constraint.
            'ModelMetadata.TrainingDataset.dataset_bucket': ['min', None],
            # d_3, d_4 match.
            'ModelMetadata.TrainingDataset.response_transform_type': 252
        }
        check(self.client_wrapper, filter_dict, 'get', 'metadata', 200,
              match_list=[d_3], non_match_list=[d_1, d_2, d_4])

    def test_metadata_success_metadata_min_metrics_filter(self):
        (model_1, model_2, model_3, model_4) = save_dicts_3(self.client_wrapper)
        # Test extremum of metadata with a metrics filter.
        filter_dict = {
            # Matches metrics 10-11 (models 3-4)
            'ModelMetrics.PredictionRuns.PredictionResults.roc_auc_score':
                ['>', 90],
            # Matches model_1, but given above constraint, matches model_3
            'ModelMetadata.TrainingDataset.dataset_oid': ['min', None]
        }
        check(self.client_wrapper, filter_dict, 'get', 'metadata', 200,
              match_list=[model_3], non_match_list=[model_1, model_2, model_4])

    def test_metadata_success_metrics_max_metadata_filter(self):
        (model_1, model_2, model_3, model_4) = save_dicts_3(self.client_wrapper)
        # Test extremum of metrics with a metadata filter.
        # Also test PredictionRuns extremum.
        filter_dict = {
            # Match metrics_10 (model_4), but given below constraint,
            # match metrics_9 (model_3)
            'ModelMetrics.PredictionRuns.PredictionResults.roc_auc_score':
                ['max', None],
            'ModelMetadata.TrainingDataset.dataset_oid': 2,  # Match models 2,3
        }
        check(self.client_wrapper, filter_dict, 'get', 'metadata', 200,
              match_list=[model_3], non_match_list=[model_1, model_2, model_4])
        
    def test_metadata_use_case_1(self):
        (model_1, model_2, model_3, model_4) = save_dicts_3(self.client_wrapper)
        # Use case: pull "all plasma protein binding models where species = dog"
        filter_dict = {
            'ModelMetadata.TrainingDataset.DatasetMetadata.assay_category':
                'plasma_protein_binding', # Match models 1,3
            'ModelMetadata.TrainingDataset.DatasetMetadata.species':
                'dog',  # Match models 2,3
        }
        check(self.client_wrapper, filter_dict, 'get', 'metadata', 200,
              match_list=[model_3], non_match_list=[model_1, model_2, model_4])

    def test_metadata_use_case_2(self):
        (model_1, model_2, model_3, model_4) = save_dicts_3(self.client_wrapper)
        # Use case: pull "all plasma protein binding models with R^2 > 0.8"
        filter_dict = {
            'ModelMetadata.TrainingDataset.DatasetMetadata.assay_category':
                'plasma_protein_binding', # Match models 1,3
            # Match metrics_6 (model_3), metrics_7 (model_4),
            # metrics_8 (model_4)
            'ModelMetrics.TrainingRun.PredictionResults.r2_score': ['>', 0.8],
        }
        check(self.client_wrapper, filter_dict, 'get', 'metadata', 200,
              match_list=[model_3], non_match_list=[model_1, model_2, model_4])

    def test_metadata_use_case_3(self):
        (model_1, model_2, model_3, model_4) = save_dicts_3(self.client_wrapper)
        # Use case: pull best model trained on specific dataset
        # (e.g., dataset_oid = xxxx)
        filter_dict = {
            'ModelMetrics.TrainingRun.PredictionResults.roc_auc_score':
                ['max', None], # Match metrics_6 (model_3)
            'ModelMetadata.TrainingDataset.dataset_oid' : 2,  # Match models 2,3
        }
        check(self.client_wrapper, filter_dict, 'get', 'metadata', 200,
              match_list=[model_3], non_match_list=[model_1, model_2, model_4])

    def test_metadata_use_case_4(self):
        (model_1, model_2, model_3, model_4) = save_dicts_3(self.client_wrapper)
        # Use case: give me the ECFP featurized NN model trained on dataset
        # with {} metadata that had the highest ROC AUC for {} dataset
        filter_dict = {
            # Match models 3,4
            'ModelMetadata.ModelParameters.featurizer': 'ECFP',
            # Match models 1,3
            'ModelMetadata.ModelParameters.model_type': 'NN',
            # Match metrics_6 (model_3)
            'ModelMetrics.TrainingRun.PredictionResults.roc_auc_score':
                ['max', None],
            # Match models 2,3
            'ModelMetadata.TrainingDataset.dataset_oid': 2,
        }
        check(self.client_wrapper, filter_dict, 'get', 'metadata', 200,
              match_list=[model_3], non_match_list=[model_1, model_2, model_4])

    def test_metadata_use_case_5(self):
        (model_1, model_2, model_3, model_4) = save_dicts_3(self.client_wrapper)
        # Use case: give me the model with ECFP features with predictions on
        # datasets with {} metadata
        filter_dict = {
            # Match models 3,4
            'ModelMetadata.ModelParameters.featurizer': 'ECFP',
        }
        check(self.client_wrapper, filter_dict, 'get', 'metadata', 200,
              match_list=[model_3, model_4], non_match_list=[model_1, model_2])


def save_dicts_1(client_wrapper):
    d = {
        'model_uuid': 'uuid_1',
        'ModelMetadata': {
            'TrainingDataset': {
                'dataset_key': 2,
                'dataset_bucket': 3,
                'dataset_oid': 2,
                'class_names': 2,
                'num_classes': 1,
                'feature_transform_type': 2,
                'response_transform_type': 2,
                'id_col': 2
            }
        }
    }
    check(client_wrapper, d, 'save', 'metadata', 200)

    return d

def save_dicts_2(client_wrapper):
    d_1 = {
        'model_uuid': 'uuid_1',
        'ModelMetadata': {
            'TrainingDataset': {
                'dataset_key': 2,
                'dataset_bucket': 3,
                'dataset_oid': 2,
                'class_names': 2,
                'num_classes': 1,
                'feature_transform_type': 2,
                'response_transform_type': 2,
                'id_col': 2000
            }
        }
    }
    check(client_wrapper, d_1, 'save', 'metadata', 200)

    d_2 = {
        'model_uuid': 'uuid_2',
        'ModelMetadata': {
            'TrainingDataset': {
                'dataset_key': 4,
                'dataset_bucket': 5,
                'dataset_oid': 7,
                'class_names': 3,
                'num_classes': 5,
                'feature_transform_type': 52,
                'response_transform_type': 42,
                'id_col': 22
            }
        }
    }
    check(client_wrapper, d_2, 'save', 'metadata', 200)

    d_3 = {
        'model_uuid': 'uuid_3',
        'ModelMetadata': {
            'TrainingDataset': {
                'dataset_key': 52,
                'dataset_bucket': 35,
                'dataset_oid': 256,
                'class_names': 218,
                'num_classes': 1729,
                'feature_transform_type': 2048,
                'response_transform_type': 252,
                'id_col': 207
            }
        }
    }
    check(client_wrapper, d_3, 'save', 'metadata', 200)

    d_4 = {
        'model_uuid': 'uuid_4',
        'ModelMetadata': {
            'TrainingDataset': {
                'dataset_key': 240,
                'dataset_bucket': 350,
                'dataset_oid': 214,
                'class_names': 217,
                'num_classes': 15,
                'feature_transform_type': 19,
                'response_transform_type': 252,
                'id_col': 25
            }
        }
    }
    check(client_wrapper, d_4, 'save', 'metadata', 200)

    return d_1, d_2, d_3, d_4

def save_dicts_3(client_wrapper):
    # Metadata dicts
    model_1 = {
        'model_uuid': 'uuid_1',
        'ModelMetadata': {
            'TrainingDataset': {
                'dataset_oid': 1,
                'DatasetMetadata': {
                    'assay_category': 'plasma_protein_binding'
                }
            },
            'ModelParameters': {
                'model_type': 'NN',
                'featurizer': 'GraphConv',
            }
        }
    }
    check(client_wrapper, model_1, 'save', 'metadata', 200)
    model_2 = {
        'model_uuid': 'uuid_2',
        'ModelMetadata': {
            'TrainingDataset': {
                'dataset_oid': 2,
                'DatasetMetadata': {
                    'species': 'dog'
                }
            },
            'ModelParameters': {
                'model_type': 'RF',
                'featurizer': 'GraphConv',
            }
        }
    }
    check(client_wrapper, model_2, 'save', 'metadata', 200)
    model_3 = {
        'model_uuid': 'uuid_3',
        'ModelMetadata': {
            'TrainingDataset': {
                'dataset_oid': 2,
                'DatasetMetadata': {
                    'assay_category': 'plasma_protein_binding',
                    'species': 'dog'
                }
            },
            'ModelParameters': {
                'model_type': 'NN',
                'featurizer': 'ECFP',
            }
        }
    }
    check(client_wrapper, model_3, 'save', 'metadata', 200)
    model_4 = {
        'model_uuid': 'uuid_4',
        'ModelMetadata': {
            'TrainingDataset': {
                'dataset_oid': 3
            },
            'ModelParameters': {
                'model_type': 'RF',
                'featurizer': 'ECFP',
            }
        }
    }
    check(client_wrapper, model_4, 'save', 'metadata', 200)
    # Metrics dicts
    metrics_1 = {
        'model_uuid': 'uuid_1',
        'ModelMetrics': {
            'TrainingRun': {
                'PredictionResults': {
                    'r2_score': 0.1,
                    'roc_auc_score': 10
                }
            }
        }
    }
    check(client_wrapper, metrics_1, 'save', 'metrics', 200)
    metrics_2 = {
        'model_uuid': 'uuid_1',
        'ModelMetrics': {
            'TrainingRun': {
                'PredictionResults': {
                    'r2_score': 0.3,
                    'roc_auc_score': 20
                }
            }
        }
    }
    check(client_wrapper, metrics_2, 'save', 'metrics', 200)
    metrics_3 = {
        'model_uuid': 'uuid_2',
        'ModelMetrics': {
            'TrainingRun': {
                'PredictionResults': {
                    'r2_score': 0.5,
                    'roc_auc_score': 30
                }
            }
        }
    }
    check(client_wrapper, metrics_3, 'save', 'metrics', 200)
    metrics_4 = {
        'model_uuid': 'uuid_2',
        'ModelMetrics': {
            'TrainingRun': {
                'PredictionResults': {
                    'r2_score': 0.7,
                    'roc_auc_score': 40
                }
            }
        }
    }
    check(client_wrapper, metrics_4, 'save', 'metrics', 200)
    metrics_5 = {
        'model_uuid': 'uuid_3',
        'ModelMetrics': {
            'TrainingRun': {
                'PredictionResults': {
                    'r2_score': 0.8,
                    'roc_auc_score': 50
                }
            }
        }
    }
    check(client_wrapper, metrics_5, 'save', 'metrics', 200)
    metrics_6 = {
        'model_uuid': 'uuid_3',
        'ModelMetrics': {
            'TrainingRun': {
                'PredictionResults': {
                    'r2_score': 0.9,
                    'roc_auc_score': 600
                }
            }
        }
    }
    check(client_wrapper, metrics_6, 'save', 'metrics', 200)
    metrics_7 = {
        'model_uuid': 'uuid_4',
        'ModelMetrics': {
            'TrainingRun': {
                'PredictionResults': {
                    'r2_score': 1.0,
                    'roc_auc_score': 70
                }
            }
        }
    }
    check(client_wrapper, metrics_7, 'save', 'metrics', 200)
    metrics_8 = {
        'model_uuid': 'uuid_4',
        'ModelMetrics': {
            'TrainingRun': {
                'PredictionResults': {
                    'r2_score': 1.1,
                    'roc_auc_score': 80
                }
            }
        }
    }
    check(client_wrapper, metrics_8, 'save', 'metrics', 200)
    metrics_9 = {
        'model_uuid': 'uuid_3',
        'ModelMetrics': {
            'PredictionRuns': {
                'PredictionResults': {
                    'r2_score': 1.1,
                    'roc_auc_score': 80
                }
            }
        }
    }
    check(client_wrapper, metrics_9, 'save', 'metrics', 200)
    metrics_10 = {
        'model_uuid': 'uuid_4',
        'ModelMetrics': {
            'PredictionRuns': {
                'PredictionResults': {
                    'r2_score': 70,
                    'roc_auc_score': 800
                }
            }
        }
    }
    check(client_wrapper, metrics_10, 'save', 'metrics', 200)
    metrics_11 = {
        'model_uuid': 'uuid_3',
        'ModelMetrics': {
            'PredictionRuns': {
                'PredictionResults': {
                    'r2_score': 70,
                    'roc_auc_score': 100
                }
            }
        }
    }
    check(client_wrapper, metrics_11, 'save', 'metrics', 200)

    return model_1, model_2, model_3, model_4
    
# https://docs.pytest.org/en/latest/getting-started.html for more on test
# classes.
class TestMetadataFailure(object):
    # https://docs.pytest.org/en/latest/xunit_setup.html for more on setup
    # methods.
    def setup_method(self):
        setup(self)

    def teardown_method(self):
        teardown(self)
        
    def test_metadata_failure_missing_uuid(self):
        d = {
            'time_built': '2018-11-06',
            'ModelMetadata': {}
        }
        errors = ('Status400Exception: model_metadata_dict does not contain'
                  ' the key model_uuid.')
        check(self.client_wrapper, d, 'save', 'metadata', 400,
              expected_errors=errors)
        
    def test_metadata_failure_non_unique_uuid(self):        
        d_1 = {
            'time_built': '2018-11-06',
            'model_uuid': 'uuid_1',
            'ModelMetadata': {}
        }
        check(self.client_wrapper, d_1, 'save', 'metadata', 200)
        
        d_2 = {
            'time_built': '2018-11-08',
            'model_uuid': 'uuid_1',
            'ModelMetadata': {}
        }
        errors = 'Status400Exception: Non-unique model_uuid=uuid_1.'
        check(self.client_wrapper, d_2, 'save', 'metadata', 400,
              expected_errors=errors)

    def test_metadata_failure_missing_metadata(self):
        d = {
            'time_built': '2018-11-06',
            'model_uuid': 'uuid_1'
        }
        errors = ('Status400Exception: model_metadata_dict does not'
                  ' contain the subdict ModelMetadata.')
        check(self.client_wrapper, d, 'save', 'metadata', 400,
              expected_errors=errors)
        
    def test_metadata_failure_disallow_collection(self):
        d = {
            'time_built': '2018-11-06',
            'model_uuid': 'uuid_1',
            'ModelMetadata': {}
        }
        collection_name = COLLECTION_NAME + '_metrics'
        errors = ('Status400Exception: collection_name={collection_name}'
                  ' contains the word "metrics".').format(
            collection_name=collection_name)
        check(self.client_wrapper, d, 'save', 'metadata', 400,
              collection_name=collection_name, expected_errors=errors)
        
    def test_metadata_failure_internal_error_on_save(self):
        d = {
            'test_internal_error': True,
            'model_uuid': 'uuid_1'
        }
        errors = 'Status500Exception: Testing internal error.'
        check(self.client_wrapper, d, 'save', 'metadata', 500,
              expected_errors=errors)
        
    def test_metadata_failure_internal_error_on_get(self):
        d = {
            'time_built': '2018-11-06',
            'model_uuid': 'uuid_1',
            'ModelMetadata': {
                'ModelSpecific': {
                    'model_type': 'NN',
                    'num_tasks': 5,
                    'uncertainty': True
                }
            }
        }
        check(self.client_wrapper, d, 'save', 'metadata', 200)
        
        filter_dict = {
            'test_internal_error' : True,
            'ModelMetadata.ModelSpecific.num_tasks' : ['=', 5]
        }
        errors = 'Status500Exception: Testing internal error.'
        check(self.client_wrapper, filter_dict, 'get', 'metadata', 500,
              expected_errors=errors)
