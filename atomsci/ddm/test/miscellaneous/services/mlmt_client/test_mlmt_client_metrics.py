from utilities import *

# Run with `python3 -m pytest -s` in the directory this file is in.
# Run a specific test with `python3 -m pytest -s test_mlmt_client_metrics.py -k
# <test_name>`
# Also test using Swagger UI or by calling these functions from an IPython
# notebook.


# https://docs.pytest.org/en/latest/getting-started.html for more on test
# classes.
class TestMetricsSuccess(object):
    # https://docs.pytest.org/en/latest/xunit_setup.html for more on setup
    # methods.
    def setup_method(self):
        setup(self)

    def teardown_method(self):
        teardown(self)
        
    def test_metrics_success_simple_syntax(self):
        (metrics_1, metrics_2, metrics_3) = save_dicts_1(self.client_wrapper)
        # Test simple syntax.
        filter_dict = {
            # Only metadata_1 matches this. metrics_1 and metrics_2 should
            # match.
            'ModelMetadata.ModelSpecific.model_type': 'NN',
            # metrics_1 and metrics_3 should match.
            'ModelMetrics.TrainingRun.PredictionResults.r2_score': 3
        }
        check(self.client_wrapper, filter_dict, 'get', 'metrics', 200,
              match_list=[metrics_1], non_match_list=[metrics_2, metrics_3])

    def test_metrics_success_primary_syntax(self):
        (metrics_1, metrics_2, metrics_3) = save_dicts_1(self.client_wrapper)
        # Test primary syntax.
        filter_dict = {
            # Only metadata_1 matches this. metrics_1 and metrics_2 should
            # match.
            'ModelMetadata.ModelSpecific.model_type': ['in', ['NN']],
            # metrics_1 and metrics_3 should match.
            'ModelMetrics.TrainingRun.PredictionResults.r2_score': ['<', 10]
        }
        check(self.client_wrapper, filter_dict, 'get', 'metrics', 200,
              match_list=[metrics_1], non_match_list=[metrics_2, metrics_3])

    def test_metrics_success_mongo_syntax(self):
        (metrics_1, metrics_2, metrics_3) = save_dicts_1(self.client_wrapper)
        # Test Mongo syntax.
        filter_dict = {
            # Only metadata_1 matches this. metrics_1 and metrics_2 should
            # match.
            'ModelMetadata.ModelSpecific.model_type': {'$nin': ['RF']},
            # metrics_1 and metrics_3 should match.
            'ModelMetrics.TrainingRun.PredictionResults.r2_score': {'$ne': 14}
        }
        check(self.client_wrapper, filter_dict, 'get', 'metrics', 200,
              match_list=[metrics_1], non_match_list=[metrics_2, metrics_3])

    def test_metrics_success_non_existent_key(self):
        (metrics_1, metrics_2, metrics_3) = save_dicts_1(self.client_wrapper)
        # Test non-existent key.
        filter_dict = {'ModelMetadata.ModelSpecific.not_a_key' : 5}
        check(self.client_wrapper, filter_dict, 'get', 'metrics', 200,
              non_match_list=[metrics_1, metrics_2, metrics_3])
        
    def test_metrics_success_min_one_key(self):
        (d_1, d_2, d_3, d_4, d_5, d_6) = save_dicts_2(self.client_wrapper)
        # Test minimum on one key.
        filter_dict = {'ModelMetrics.TrainingRun.PredictionResults.r2_score':
                           ['min', None]}
        check(self.client_wrapper, filter_dict, 'get', 'metrics', 200,
              match_list=[d_1], non_match_list=[d_2, d_3, d_4])

    def test_metrics_success_max_one_key(self):
        (d_1, d_2, d_3, d_4, d_5, d_6) = save_dicts_2(self.client_wrapper)
        # Test maximum on one key.
        filter_dict = {'ModelMetrics.TrainingRun.PredictionResults.r2_score':
                           ['max', None]}
        check(self.client_wrapper, filter_dict, 'get', 'metrics', 200,
              match_list=[d_4], non_match_list=[d_1, d_2, d_3])

    def test_metrics_success_extrema_diff_keys_same_doc(self):
        (d_1, d_2, d_3, d_4, d_5, d_6) = save_dicts_2(self.client_wrapper)
        # Test minimum and maximum on different keys
        filter_dict = {
            'ModelMetrics.TrainingRun.PredictionResults.r2_score':
                ['min', 1],
            'ModelMetrics.TrainingRun.PredictionResults.task_r2_scores':
                ['max', 2]
        }
        check(self.client_wrapper, filter_dict, 'get', 'metrics', 200,
              match_list=[d_1], non_match_list=[d_2, d_3, d_4])

    def test_metrics_success_extrema_diff_keys_diff_docs(self):
        (d_1, d_2, d_3, d_4, d_5, d_6) = save_dicts_2(self.client_wrapper)
        # Test minimum and maximum on different keys.
        filter_dict = {
            'ModelMetrics.TrainingRun.PredictionResults.r2_score':
                ['min', 1],  # d_1 is higher priority and will therefore match.
            'ModelMetrics.TrainingRun.PredictionResults.matthews_cc':
                ['max', 2]  # d_3
        }
        check(self.client_wrapper, filter_dict, 'get', 'metrics', 200,
              match_list=[d_1], non_match_list=[d_2, d_3, d_4])

    def test_metrics_success_max_two_matches(self):
        (d_1, d_2, d_3, d_4, d_5, d_6) = save_dicts_2(self.client_wrapper)
        # Test returning two dicts for maximum.
        filter_dict = {'ModelMetrics.TrainingRun.PredictionResults.kappa':
                       ['max', None]}
        check(self.client_wrapper, filter_dict, 'get', 'metrics', 200,
              match_list=[d_3, d_4], non_match_list=[d_1, d_2])

    def test_metrics_success_metrics_max_metadata_filter(self):
        (d_1, d_2, d_3, d_4, d_5, d_6) = save_dicts_2(self.client_wrapper)
        # Test metrics max with a metadata filter
        filter_dict = {
            'ModelMetadata.ModelSpecific.num_classes': 2,
            'ModelMetrics.TrainingRun.PredictionResults.kappa': ['max', None]
        }
        check(self.client_wrapper, filter_dict, 'get', 'metrics', 200,
              match_list=[d_3], non_match_list=[d_1, d_2, d_4])

    def test_metrics_success_extrema_metadata_metrics(self):
        (d_1, d_2, d_3, d_4, d_5, d_6) = save_dicts_2(self.client_wrapper)
        # Test extremum on both metadata and metrics
        filter_dict = {
            # Matches model 1 (d_1, d_3, d_5)
            'ModelMetadata.ModelSpecific.num_classes': ['min', 1],
            # Matches d_6, but with above constraint matches d_3 instead.
            'ModelMetrics.TrainingRun.PredictionResults.kappa': ['max', 2]
        }
        check(self.client_wrapper, filter_dict, 'get', 'metrics', 200,
              match_list=[d_3], non_match_list=[d_1, d_2, d_4])

    def test_metrics_success_max_extra_filter(self):
        (d_1, d_2, d_3, d_4, d_5, d_6) = save_dicts_2(self.client_wrapper)
        # Test max with extra filter
        filter_dict = {
            'ModelMetrics.TrainingRun.PredictionResults.r2_score':
                ['min', None],
            'ModelMetrics.TrainingRun.PredictionResults.prc_auc_score': 1
        }
        check(self.client_wrapper, filter_dict, 'get', 'metrics', 200,
              match_list=[d_1], non_match_list=[d_2, d_3, d_4])

    def test_metrics_success_min_two_keys(self):
        (d_1, d_2, d_3, d_4, d_5, d_6) = save_dicts_2(self.client_wrapper)
        # Test minimum on two different keys.
        filter_dict = {
            'ModelMetrics.TrainingRun.PredictionResults.r2_score':
                ['min', 1],
            'ModelMetrics.TrainingRun.PredictionResults.rms_score':
                ['min', 2]
        }
        check(self.client_wrapper, filter_dict, 'get', 'metrics', 200,
              match_list=[d_1], non_match_list=[d_2, d_3, d_4])

    def test_metrics_success_min_non_existent_key(self):
        (d_1, d_2, d_3, d_4, d_5, d_6) = save_dicts_2(self.client_wrapper)
        # Test minimum on non_existent key
        filter_dict = {'ModelMetrics.TrainingRun.PredictionResults.not_a_key':
                           ['min', None]}
        check(self.client_wrapper, filter_dict, 'get', 'metrics', 200,
              match_list=[], non_match_list=[d_1, d_2, d_3, d_4])

    def test_metrics_success_min_subset(self):
        (d_1, d_2, d_3, d_4, d_5, d_6) = save_dicts_2(self.client_wrapper)
        # Test minimum for a subset.
        filter_dict = {
            # d_1 matches, but d_3 matches with the following constraint.
            'ModelMetrics.TrainingRun.PredictionResults.r2_score':
                ['min', None],
            # d_3, d_4 match.
            'ModelMetrics.TrainingRun.PredictionResults.kappa': 252
        }
        check(self.client_wrapper, filter_dict, 'get', 'metrics', 200,
              match_list=[d_3], non_match_list=[d_1, d_2, d_4])

    def test_metrics_success_no_filters(self):
        (d_1, d_2, d_3, d_4, d_5, d_6) = save_dicts_2(self.client_wrapper)
        # Test no filters.
        check(self.client_wrapper, {}, 'get', 'metrics', 200,
              match_list=[d_1, d_2, d_3, d_4])

    def test_metrics_success_metadata_min_metrics_filter(self):
        (d_1, d_2, d_3, d_4, d_5, d_6) = save_dicts_2(self.client_wrapper)
        # Test metadata min with metrics filter.
        filter_dict = {
            # Matches model 1 (metrics 1,3,5) -
            # only d_3 given below constraints.
            'ModelMetadata.ModelSpecific.num_classes': ['min', None],
            # Matches d_3, d_4
            'ModelMetrics.TrainingRun.PredictionResults.kappa': 252
        }
        check(self.client_wrapper, filter_dict, 'get', 'metrics', 200,
              match_list=[d_3], non_match_list=[d_1, d_2, d_4, d_5, d_6])

    def test_metrics_success_PredictionRuns_max(self):
        (d_1, d_2, d_3, d_4, d_5, d_6) = save_dicts_2(self.client_wrapper)
        # Test metrics max with metadata filter.
        # Also test PredictionRuns max with a metadata filter.
        filter_dict = {
            'ModelMetadata.ModelSpecific.num_classes': 2,
            # d_6 matches, but given the above constraint, d_5 matches
            'ModelMetrics.PredictionRuns.PredictionResults.kappa': ['max', None]
        }
        check(self.client_wrapper, filter_dict, 'get', 'metrics', 200,
              match_list=[d_5], non_match_list=[d_1, d_2, d_3, d_4, d_6])
        
    def test_metrics_success_comprehensive_simple_syntax(self):
        (metrics_1, metrics_2, metrics_3a, metrics_3b, metrics_3c,
         metrics_4a, metrics_4b, metrics_4c,
         metrics_5a, metrics_5b, metrics_5c,
         metrics_5d, metrics_5e, metrics_5f) = save_dicts_3(self.client_wrapper)
        # Test simple syntax.
        filter_dict = {
            # Only metadata_1 matches this. metrics_1,2,5a-f should match.
            'ModelMetadata.ModelSpecific.model_type': 'NN',
            # metrics_1,3a,5a should match.
            'ModelMetrics.TrainingRun.PredictionResults.r2_score': 3
        }
        check(self.client_wrapper, filter_dict, 'get', 'metrics', 200,
              match_list=[metrics_1, metrics_5a], asserting_all_matches=True)

    def test_metrics_success_comprehensive_primary_syntax(self):
        (metrics_1, metrics_2, metrics_3a, metrics_3b, metrics_3c,
         metrics_4a, metrics_4b, metrics_4c,
         metrics_5a, metrics_5b, metrics_5c,
         metrics_5d, metrics_5e, metrics_5f) = save_dicts_3(self.client_wrapper)
        # Test primary syntax.
        filter_dict = {
            # Only metadata_1 matches this. metrics_1,2,5a-f should match.
            'ModelMetadata.ModelSpecific.model_type': ['in', ['NN']],
            # metrics_1,3a,5a should match.
            'ModelMetrics.TrainingRun.PredictionResults.r2_score': ['<', 10]
        }
        check(self.client_wrapper, filter_dict, 'get', 'metrics', 200,
              match_list=[metrics_1, metrics_5a], asserting_all_matches=True)

    def test_metrics_success_comprehensive_mongo_syntax(self):
        (metrics_1, metrics_2, metrics_3a, metrics_3b, metrics_3c,
         metrics_4a, metrics_4b, metrics_4c,
         metrics_5a, metrics_5b, metrics_5c,
         metrics_5d, metrics_5e, metrics_5f) = save_dicts_3(self.client_wrapper)
        # Test Mongo syntax.
        filter_dict = {
            # Only metadata_1 matches this. metrics_1,2,5a-f should match.
            'ModelMetadata.ModelSpecific.model_type': {'$nin': ['RF']},
            # metrics_1,3a-c,4a-c,5a-f should match.
            'ModelMetrics.PredictionRuns.PredictionResults.r2_score':
                {'$ne': 14}
        }
        check(self.client_wrapper, filter_dict, 'get', 'metrics', 200,
              match_list=[metrics_1, metrics_5a, metrics_5b, metrics_5c, metrics_5d, metrics_5e, metrics_5f],
              asserting_all_matches=True)

    def test_metrics_success_comprehensive_non_existent_key(self):
        (metrics_1, metrics_2, metrics_3a, metrics_3b, metrics_3c,
         metrics_4a, metrics_4b, metrics_4c,
         metrics_5a, metrics_5b, metrics_5c,
         metrics_5d, metrics_5e, metrics_5f) = save_dicts_3(self.client_wrapper)
        # Test non-existent key.
        filter_dict = {'ModelMetadata.ModelSpecific.not_a_key' : 5}
        check(self.client_wrapper, filter_dict, 'get', 'metrics', 200,
              asserting_all_matches=True)

    def test_metrics_success_TrainingRun_max(self):
        (metrics_1, metrics_2, metrics_3a, metrics_3b, metrics_3c,
         metrics_4a, metrics_4b, metrics_4c,
         metrics_5a, metrics_5b, metrics_5c,
         metrics_5d, metrics_5e, metrics_5f) = save_dicts_3(self.client_wrapper)
        # Test extremum on TrainingRun.
        filter_dict = {'ModelMetrics.TrainingRun.PredictionResults.r2_score':
                           ['max', None]}
        check(self.client_wrapper, filter_dict, 'get', 'metrics', 200,
              match_list=[metrics_5c], asserting_all_matches=True)

    def test_metrics_success_PredictionRuns_max(self):
        (metrics_1, metrics_2, metrics_3a, metrics_3b, metrics_3c,
         metrics_4a, metrics_4b, metrics_4c,
         metrics_5a, metrics_5b, metrics_5c,
         metrics_5d, metrics_5e, metrics_5f) = save_dicts_3(self.client_wrapper)
        # Test extremum on PredictionRuns.
        filter_dict = \
            {'ModelMetrics.PredictionRuns.PredictionResults.num_compounds':
                 ['max', None]}
        check(self.client_wrapper, filter_dict, 'get', 'metrics', 200,
              match_list=[metrics_5f], asserting_all_matches=True)
        
    def test_metrics_use_cases(self):
        (metrics_1, metrics_2, metrics_3, metrics_4, metrics_5, metrics_6,
         metrics_7, metrics_8) = save_dicts_4(self.client_wrapper)
        # Use case: show performance of all graph convolution models tested on
        # solubility datasets.
        # (give me all the results for models on datasets with these attributes)
        filter_dict = {
            # Match models 1,2,3.
            'ModelMetadata.ModelParameters.featurizer' : 'GraphConv',
        }
        check(self.client_wrapper, filter_dict, 'get', 'metrics', 200,
              match_list=[metrics_1, metrics_2, metrics_5, metrics_6],
              non_match_list=[metrics_3, metrics_4, metrics_7, metrics_8])


def save_dicts_1(client_wrapper):
    # Save two metadata dicts.
    metadata_1 = {
        'time_built': '2018-11-06',
        'model_uuid': 'uuid_1',
        'ModelMetadata': {
            'ModelSpecific': {
                'model_type': 'NN',
                'num_tasks': 3,
                'uncertainty': True
            }
        }
    }
    check(client_wrapper, metadata_1, 'save', 'metadata', 200)
    metadata_2 = {
        'time_built': '2018-11-06',
        'model_uuid': 'uuid_2',
        'ModelMetadata': {
            'ModelSpecific': {
                'model_type': 'RF',
                'num_tasks': 3,
                'uncertainty': False
            }
        }
    }
    check(client_wrapper, metadata_2, 'save', 'metadata', 200)
    # Save three metrics dicts.
    metrics_1 = {
        'model_uuid': 'uuid_1',
        'ModelMetrics': {
            'TrainingRun': {
                'PredictionResults': {
                    'r2_score': 3,
                    'num_compounds': 5
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
                    'r2_score': 14,
                    'num_compounds': 7
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
                    'r2_score': 3,
                    'num_compounds': 5
                }
            }
        }
    }
    check(client_wrapper, metrics_3, 'save', 'metrics', 200)
    return metrics_1, metrics_2, metrics_3

def save_dicts_2(client_wrapper):
    # Save model dicts.
    model_1 = {
        'model_uuid': 'uuid_1',
        'ModelMetadata': {
            'ModelSpecific': {
                'num_classes': 2
            }
        }
    }
    check(client_wrapper, model_1, 'save', 'metadata', 200)
    model_2 = {
        'model_uuid': 'uuid_2',
        'ModelMetadata': {
            'ModelSpecific': {
                'num_classes': 4
            }
        }
    }
    check(client_wrapper, model_2, 'save', 'metadata', 200)
    # Save metrics dicts.
    d_1 = {
        'model_uuid': 'uuid_1',
        'ModelMetrics': {
            'TrainingRun': {
                'PredictionResults': {
                    'r2_score': 2,
                    'rms_score': 3,
                    'mae_score': 2,
                    'roc_auc_score': 2,
                    'prc_auc_score': 1,
                    'matthews_cc': 2,
                    'kappa': 2,
                    'task_r2_scores': 2000
                }
            }
        }
    }
    check(client_wrapper, d_1, 'save', 'metrics', 200)
    d_2 = {
        'model_uuid': 'uuid_2',
        'ModelMetrics': {
            'TrainingRun': {
                'PredictionResults': {
                    'r2_score': 4,
                    'rms_score': 5,
                    'mae_score': 7,
                    'roc_auc_score': 3,
                    'prc_auc_score': 5,
                    'matthews_cc': 52,
                    'kappa': 42,
                    'task_r2_scores': 22
                }
            }
        }
    }
    check(client_wrapper, d_2, 'save', 'metrics', 200)
    d_3 = {
        'model_uuid': 'uuid_1',
        'ModelMetrics': {
            'TrainingRun': {
                'PredictionResults': {
                    'r2_score': 52,
                    'rms_score': 35,
                    'mae_score': 256,
                    'roc_auc_score': 218,
                    'prc_auc_score': 1729,
                    'matthews_cc': 2048,
                    'kappa': 252,
                    'task_r2_scores': 207
                }
            }
        }
    }
    check(client_wrapper, d_3, 'save', 'metrics', 200)
    d_4 = {
        'model_uuid': 'uuid_2',
        'ModelMetrics': {
            'TrainingRun': {
                'PredictionResults': {
                    'r2_score': 240,
                    'rms_score': 350,
                    'mae_score': 214,
                    'roc_auc_score': 217,
                    'prc_auc_score': 15,
                    'matthews_cc': 19,
                    'kappa': 252,
                    'task_r2_scores': 25
                }
            }
        }
    }
    check(client_wrapper, d_4, 'save', 'metrics', 200)
    d_5 = {
        'model_uuid': 'uuid_1',
        'ModelMetrics': {
            'PredictionRuns': {
                'PredictionResults': {
                    'r2_score': 240,
                    'rms_score': 350,
                    'mae_score': 214,
                    'roc_auc_score': 217,
                    'prc_auc_score': 15,
                    'matthews_cc': 19,
                    'kappa': 250,
                    'task_r2_scores': 25
                }
            }
        }
    }
    check(client_wrapper, d_5, 'save', 'metrics', 200)
    d_6 = {
        'model_uuid': 'uuid_2',
        'ModelMetrics': {
            'PredictionRuns': {
                'PredictionResults': {
                    'r2_score': 240,
                    'rms_score': 350,
                    'mae_score': 214,
                    'roc_auc_score': 217,
                    'prc_auc_score': 15,
                    'matthews_cc': 19,
                    'kappa': 700,
                    'task_r2_scores': 25
                }
            }
        }
    }
    check(client_wrapper, d_6, 'save', 'metrics', 200)
    return d_1, d_2, d_3, d_4, d_5, d_6

def save_dicts_3(client_wrapper):
    # Save two metadata dicts.
    metadata_1 = {
        'time_built': '2018-11-06',
        'model_uuid': 'uuid_1',
        'ModelMetadata': {
            'ModelSpecific': {
                'model_type': 'NN',
                'num_tasks': 3,
                'uncertainty': True
            }
        }
    }
    check(client_wrapper, metadata_1, 'save', 'metadata', 200)
    metadata_2 = {
        'time_built': '2018-11-06',
        'model_uuid': 'uuid_2',
        'ModelMetadata': {
            'ModelSpecific': {
                'model_type': 'RF',
                'num_tasks': 3,
                'uncertainty': False
            }
        }
    }
    check(client_wrapper, metadata_2, 'save', 'metadata', 200)
    # Save metrics dicts.
    # Single TrainingRun.
    metrics_1 = {
        'model_uuid': 'uuid_1',
        'ModelMetrics': {
            'TrainingRun': {
                'PredictionResults': {
                    'r2_score': 3,
                    'num_compounds': 5
                }
            }
        }
    }
    check(client_wrapper, metrics_1, 'save', 'metrics', 200)
    # Single PredictionRuns.
    metrics_2 = {
        'model_uuid': 'uuid_1',
        'ModelMetrics': {
            'PredictionRuns': {
                'PredictionResults': {
                    'r2_score': 14,
                    'num_compounds': 7
                }
            }
        }
    }
    check(client_wrapper, metrics_2, 'save', 'metrics', 200)
    # List of TraningRun.
    metrics_3 = {
        'model_uuid': 'uuid_2',
        'ModelMetrics': {
            'TrainingRun': [
                {
                    'PredictionResults': {
                        'r2_score': 3,
                        'num_compounds': 5
                    }
                },
                {
                    'PredictionResults': {
                        'r2_score': 72,
                        'num_compounds': 57
                    }
                },
                {
                    'PredictionResults': {
                        'r2_score': 76,
                        'num_compounds': 34
                    }
                }
            ]
        }
    }
    check(client_wrapper, metrics_3, 'save', 'metrics', 200)
    # List of PredictionRun.
    metrics_4 = {
        'model_uuid': 'uuid_2',
        'ModelMetrics': {
            'PredictionRuns': [
                {
                    'PredictionResults': {
                        'r2_score': 3,
                        'num_compounds': 5
                    }
                },
                {
                    'PredictionResults': {
                        'r2_score': 72,
                        'num_compounds': 57
                    }
                },
                {
                    'PredictionResults': {
                        'r2_score': 76,
                        'num_compounds': 34
                    }
                }
            ]
        }
    }
    check(client_wrapper, metrics_4, 'save', 'metrics', 200)
    # Lists of TraningRun and PredictionRuns.
    metrics_5 = {
        'model_uuid': 'uuid_1',
        'ModelMetrics': {
            'TrainingRun': [
                {
                    'PredictionResults': {
                        'r2_score': 3,
                        'num_compounds': 5
                    }
                },
                {
                    'PredictionResults': {
                        'r2_score': 72,
                        'num_compounds': 57
                    }
                },
                {
                    'PredictionResults': {
                        'r2_score': 176,
                        'num_compounds': 34
                    }
                }
            ],
            'PredictionRuns': [
                {
                    'PredictionResults': {
                        'r2_score': 3,
                        'num_compounds': 5
                    }
                },
                {
                    'PredictionResults': {
                        'r2_score': 72,
                        'num_compounds': 57
                    }
                },
                {
                    'PredictionResults': {
                        'r2_score': 76,
                        'num_compounds': 345
                    }
                }
            ]
        }
    }
    check(client_wrapper, metrics_5, 'save', 'metrics', 200)
    # Name sub-dicts
    metrics_3a = {
        'model_uuid': 'uuid_2',
        'ModelMetrics': {
            'TrainingRun': metrics_3['ModelMetrics']['TrainingRun'][0]
        }
    }
    metrics_3b = {
        'model_uuid': 'uuid_2',
        'ModelMetrics': {
            'TrainingRun': metrics_3['ModelMetrics']['TrainingRun'][1]
        }
    }
    metrics_3c = {
        'model_uuid': 'uuid_2',
        'ModelMetrics': {
            'TrainingRun': metrics_3['ModelMetrics']['TrainingRun'][2]
        }
    }
    metrics_4a = {
        'model_uuid': 'uuid_2',
        'ModelMetrics': {
            'PredictionRuns': metrics_4['ModelMetrics']['PredictionRuns'][0]
        }
    }
    metrics_4b = {
        'model_uuid': 'uuid_2',
        'ModelMetrics': {
            'PredictionRuns': metrics_4['ModelMetrics']['PredictionRuns'][1]
        }
    }
    metrics_4c = {
        'model_uuid': 'uuid_2',
        'ModelMetrics': {
            'PredictionRuns': metrics_4['ModelMetrics']['PredictionRuns'][2]
        }
    }
    metrics_5a = {
        'model_uuid': 'uuid_1',
        'ModelMetrics': {
            'TrainingRun': metrics_5['ModelMetrics']['TrainingRun'][0]
        }
    }
    metrics_5b = {
        'model_uuid': 'uuid_1',
        'ModelMetrics': {
            'TrainingRun': metrics_5['ModelMetrics']['TrainingRun'][1]
        }
    }
    metrics_5c = {
        'model_uuid': 'uuid_1',
        'ModelMetrics': {
            'TrainingRun': metrics_5['ModelMetrics']['TrainingRun'][2]
        }
    }
    metrics_5d = {
        'model_uuid': 'uuid_1',
        'ModelMetrics': {
            'PredictionRuns': metrics_5['ModelMetrics']['PredictionRuns'][0]
        }
    }
    metrics_5e = {
        'model_uuid': 'uuid_1',
        'ModelMetrics': {
            'PredictionRuns': metrics_5['ModelMetrics']['PredictionRuns'][1]
        }
    }
    metrics_5f = {
        'model_uuid': 'uuid_1',
        'ModelMetrics': {
            'PredictionRuns': metrics_5['ModelMetrics']['PredictionRuns'][2]
        }
    }
    return (metrics_1, metrics_2, metrics_3a, metrics_3b, metrics_3c,
            metrics_4a, metrics_4b, metrics_4c,
            metrics_5a, metrics_5b, metrics_5c,
            metrics_5d, metrics_5e, metrics_5f)

def save_dicts_4(client_wrapper):
    # Metadata dicts
    model_1 = {
        'model_uuid': 'uuid_1',
        'ModelMetadata': {
            'ModelParameters': {
                'featurizer': 'GraphConv',
            }
        }
    }
    check(client_wrapper, model_1, 'save', 'metadata', 200)
    model_2 = {
        'model_uuid': 'uuid_2',
        'ModelMetadata': {
            'ModelParameters': {
                'featurizer': 'GraphConv',
            }
        }
    }
    check(client_wrapper, model_2, 'save', 'metadata', 200)
    model_3 = {
        'model_uuid': 'uuid_3',
        'ModelMetadata': {
            'ModelParameters': {
                'featurizer': 'GraphConv',
            }
        }
    }
    check(client_wrapper, model_3, 'save', 'metadata', 200)
    model_4 = {
        'model_uuid': 'uuid_4',
        'ModelMetadata': {
            'ModelParameters': {
                'featurizer': 'Not GraphConv',
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
                    'r2_score': 1
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
                    'r2_score': 2
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
                    'r2_score': 3
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
                    'r2_score': 4
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
                    'r2_score': 5
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
                    'r2_score': 6
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
                    'r2_score': 7
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
                    'r2_score': 8
                }
            }
        }
    }
    check(client_wrapper, metrics_8, 'save', 'metrics', 200)
    return (metrics_1, metrics_2, metrics_3, metrics_4,
            metrics_5, metrics_6, metrics_7, metrics_8)


# https://docs.pytest.org/en/latest/getting-started.html for more on test
# classes.
class TestMetricsFailure(object):
    # https://docs.pytest.org/en/latest/xunit_setup.html for more on setup
    # methods.
    def setup_method(self):
        setup(self)

    def teardown_method(self):
        teardown(self)
        
    def test_metrics_failure_missing_uuid(self):
        d = {
            'ModelMetrics': {
                'TrainingRun': {}
            }
        }
        errors = ('Status400Exception: model_metrics_dict does not'
                  ' contain the key model_uuid.')
        check(self.client_wrapper, d, 'save', 'metrics', 400,
              expected_errors=errors)
        
    def test_metrics_failure_invalid_keys(self):
        d = {
            'model_uuid' : 'uuid_1',
            'ModelMetrics' : {
                'TrainingRun' : {}
            },
            'x' : 50
        }
        errors = ("Status400Exception: model_metrics_dict contains"
                  " invalid keys: ['x'].")
        check(self.client_wrapper, d, 'save', 'metrics', 400,
              expected_errors=errors)

    def test_metrics_failure_missing_metrics(self):
        d = {
            'model_uuid' : 'uuid_1'
        }
        errors = ('Status400Exception: model_metrics_dict does not'
                  ' contain the subdict ModelMetrics.')
        check(self.client_wrapper, d, 'save', 'metrics', 400,
              expected_errors=errors)
        
    def test_metrics_failure_missing_subdicts(self):
        d = {
            'model_uuid' : 'uuid_1',
            'ModelMetrics' : {}
        }
        errors = ('Status400Exception: model_metrics_dict["ModelMetrics"]'
                  ' contains neither TrainingRun nor PredictionRuns.')
        check(self.client_wrapper, d, 'save', 'metrics', 400,
              expected_errors=errors)
        
    def test_metrics_failure_invalid_subdicts(self):
        d = {
            'model_uuid' : 'uuid_1',
            'ModelMetrics' : {
                'X' : {}
            }
        }
        errors = ('Status400Exception: model_metrics_dict["ModelMetrics"]'
                  ' contains invalid keys: [\'X\'].')
        check(self.client_wrapper, d, 'save', 'metrics', 400,
              expected_errors=errors)
        
    def test_metrics_failure_disallow_collection(self):
        d = {
            'model_uuid' : 'uuid_1',
            'ModelMetrics' : {
                'TrainingRun' : {}
            }
        }
        collection_name = COLLECTION_NAME + '_metrics'
        errors = ('Status400Exception: collection_name={collection_name}'
                  ' contains the word "metrics". collection_name + "_metrics" '
                  'is automatically used to store metrics.'
                  ).format(collection_name=collection_name)
        check(self.client_wrapper, d, 'save', 'metrics', 400,
              collection_name=collection_name, expected_errors=errors)
        
    def test_metrics_failure_internal_error_on_save(self):
        d = {
            'test_internal_error': True,
            'model_uuid': 'uuid_1'
        }
        errors = 'Status500Exception: Testing internal error.'
        check(self.client_wrapper, d, 'save', 'metrics', 500,
              expected_errors=errors)
        
    def test_metrics_failure_internal_error_on_get(self):
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
        d = {
            'model_uuid': 'uuid_1',
            'ModelMetrics': {
                'TrainingRun': {
                    'PredictionResults': {
                        'r2_score': 2
                    }
                }
            }
        }
        check(self.client_wrapper, d, 'save', 'metrics', 200)
        
        filter_dict = {
            'test_internal_error': True,
            'ModelMetrics.TrainingRun.PredictionResults.r2_score': ['=', 2]
        }
        errors = 'Status500Exception: Testing internal error.'
        check(self.client_wrapper, filter_dict, 'get', 'metrics', 500,
              expected_errors=errors)
