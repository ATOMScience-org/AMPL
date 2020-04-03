from utilities import *

# Run with `python3 -m pytest -s` in the directory this file is in.
# Run a specific test with `python3 -m pytest -s test_mlmt_client_models.py -k
# <test_name>`
# Also test using Swagger UI or by calling these functions from an IPython
# notebook.

# https://docs.pytest.org/en/latest/getting-started.html for more on test
# classes.
class TestModelsSuccess(object):
    
    # https://docs.pytest.org/en/latest/xunit_setup.html for more on setup
    # methods.
    def setup_method(self):
        setup(self)

    def teardown_method(self):
        teardown(self)
        
    def test_models_success_filter_metadata_TrainingRun(self):
        save_dicts_1(self.client_wrapper)
        # Test match on both metadata and TrainingRun.
        filter_dict = {
            # Only metadata_1 matches this. metrics_1,2,4 should match.
            'ModelMetadata.ModelSpecific.model_type': 'NN',
            # metrics_1 and metrics_3 should match.
            'ModelMetrics.TrainingRun.PredictionResults.r2_score': 3
        }
        expected_match = {
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
        check(self.client_wrapper, filter_dict, 'get', 'metadata', 200,
              match_list=[expected_match])
        expected_match = {
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
        check(self.client_wrapper, filter_dict, 'get', 'metrics', 200,
              match_list=[expected_match])
        expected_match = {
            'time_built': '2018-11-06',
            'model_uuid': 'uuid_1',
            'ModelMetadata': {
                'ModelSpecific': {
                    'model_type': 'NN',
                    'num_tasks': 3,
                    'uncertainty': True
                }
            },
            'ModelMetrics': {
                'TrainingRun': [
                    {
                        'PredictionResults': {
                            'r2_score': 3,
                            'num_compounds': 5
                        }
                    }
                ],
                'PredictionRuns': []
            }
        }
        check(self.client_wrapper, filter_dict, 'get', 'model', 200,
              match_list=[expected_match])

    def test_models_success_filter_metadata_PredictionRuns(self):
        save_dicts_1(self.client_wrapper)
        # Test filter on metadata and PredictionRuns.
        filter_dict = {
            # Only metadata_1 matches this. metrics_1,2,4 should match.
            'ModelMetadata.ModelSpecific.model_type': 'NN',
            # metrics_4 should match.
            'ModelMetrics.PredictionRuns.PredictionResults.r2_score': 3
        }
        expected_match = {
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
        check(self.client_wrapper, filter_dict, 'get', 'metadata', 200,
              match_list=[expected_match])
        expected_match = {
            'model_uuid': 'uuid_1',
            'ModelMetrics': {
                'PredictionRuns': {
                    'PredictionResults': {
                        'r2_score': 3,
                        'num_compounds': 5
                    }
                }
            }
        }
        check(self.client_wrapper, filter_dict, 'get', 'metrics', 200,
              match_list=[expected_match])
        expected_match = {
            'time_built': '2018-11-06',
            'model_uuid': 'uuid_1',
            'ModelMetadata': {
                'ModelSpecific': {
                    'model_type': 'NN',
                    'num_tasks': 3,
                    'uncertainty': True
                }
            },
            'ModelMetrics': {
                'PredictionRuns': [
                    {
                        'PredictionResults': {
                            'r2_score': 3,
                            'num_compounds': 5
                        }
                    }
                ],
                'TrainingRun': []
            }
        }
        check(self.client_wrapper, filter_dict, 'get', 'model', 200,
              match_list=[expected_match])

    def test_models_success_no_match(self):
        save_dicts_1(self.client_wrapper)
        # Test no match.
        filter_dict = {
            # Only metadata_2 matches this. metrics_3 should match.
            'ModelMetadata.ModelSpecific.model_type': 'RF',
            # Only metrics_2 matches this.
            'ModelMetrics.TrainingRun.PredictionResults.r2_score': 14
        }
        check(self.client_wrapper, filter_dict, 'get', 'model', 200,
              match_list=[])

    def test_models_success_extremum(self):
        save_dicts_1(self.client_wrapper)
        # Test extremum
        filter_dict = {
            # Only metadata_1 matches this. metrics_1,2,4 should match.
            'ModelMetadata.ModelSpecific.model_type': 'NN',
            # Only metrics_2 matches this.
            'ModelMetrics.TrainingRun.PredictionResults.r2_score': ['max', None]
        }
        expected_match = {
            'time_built': '2018-11-06',
            'model_uuid': 'uuid_1',
            'ModelMetadata': {
                'ModelSpecific': {
                    'model_type': 'NN',
                    'num_tasks': 3,
                    'uncertainty': True
                }
            },
            'ModelMetrics': {
                'TrainingRun': [
                    {
                        'PredictionResults': {
                            'r2_score': 14,
                            'num_compounds': 7
                        }
                    }
                ],
                'PredictionRuns': []
            }
        }
        check(self.client_wrapper, filter_dict, 'get', 'model', 200,
              match_list=[expected_match])

    def test_models_success_regular_filters(self):
        save_dicts_2(self.client_wrapper)
        # Test regular filters.
        filter_dict = {
            'ModelMetadata.TrainingDataset.dataset_key': 'key_1',
            'ModelMetadata.TrainingDataset.bucket': 'bucket_1',
            'ModelMetrics.TrainingRun.label': 'best',
            'ModelMetrics.TrainingRun.subset': 'valid'
        }
        expected_match = {
            'time_built': '2019-02-14',
            'model_uuid': 'uuid_1',
            'ModelMetadata': {
                'TrainingDataset': {
                    'dataset_key': 'key_1',
                    'bucket': 'bucket_1'
                }
            },
            'ModelMetrics': {
                'TrainingRun': [
                    {
                        'label': 'best',
                        'subset': 'valid',
                        'PredictionResults': {
                            'r2_score': 50
                        }
                    },
                    {
                        'label': 'best',
                        'subset': 'valid',
                        'PredictionResults': {
                            'r2_score': 20
                        }
                    }
                ],
                'PredictionRuns': []
            }
        }
        check(self.client_wrapper, filter_dict, 'get', 'model', 200,
              match_list=[expected_match])

    def test_models_success_global_max(self):
        save_dicts_2(self.client_wrapper)
        # Test match on global maximum.
        filter_dict = {
            'ModelMetrics.TrainingRun.PredictionResults.r2_score': ['max', None]
        }
        expected_match = {
            'time_built': '2019-02-14',
            'model_uuid': 'uuid_1',
            'ModelMetadata': {
                'TrainingDataset': {
                    'dataset_key': 'key_1',
                    'bucket': 'bucket_1'
                }
            },
            'ModelMetrics': {
                'TrainingRun': [
                    {
                        'label': 'best',
                        'subset': 'train',
                        'PredictionResults': {
                            'r2_score': 100
                        }
                    }
                ],
                'PredictionRuns': []
            }
        }
        check(self.client_wrapper, filter_dict, 'get', 'model', 200,
              match_list=[expected_match])

    def test_models_success_valid_max(self):
        save_dicts_2(self.client_wrapper)
        # Test on maximum local to validation sets.
        filter_dict = {
            'ModelMetrics.TrainingRun.label': 'best',
            'ModelMetrics.TrainingRun.subset': 'valid',
            'ModelMetrics.TrainingRun.PredictionResults.r2_score': ['max', None]
        }
        expected_match = {
            'time_built': '2019-02-14',
            'model_uuid': 'uuid_2',
            'ModelMetadata': {
                'TrainingDataset': {
                    'dataset_key': 'key_2',
                    'bucket': 'bucket_2'
                }
            },
            'ModelMetrics': {
                'TrainingRun': [
                    {
                        'label': 'best',
                        'subset': 'valid',
                        'PredictionResults': {
                            'r2_score': 70
                        }
                    }
                ],
                'PredictionRuns': []
            }
        }
        check(self.client_wrapper, filter_dict, 'get', 'model', 200,
              match_list=[expected_match])

    def test_models_success_valid_max_subset(self):
        save_dicts_2(self.client_wrapper)
        # Test on maximum local to subset of validation sets.
        filter_dict = {
            'ModelMetadata.TrainingDataset.dataset_key': 'key_1',
            'ModelMetadata.TrainingDataset.bucket': 'bucket_1',
            'ModelMetrics.TrainingRun.label': 'best',
            'ModelMetrics.TrainingRun.subset': 'valid',
            'ModelMetrics.TrainingRun.PredictionResults.r2_score': ['max', None]
        }
        expected_match = {
            'time_built': '2019-02-14',
            'model_uuid': 'uuid_1',
            'ModelMetadata': {
                'TrainingDataset': {
                    'dataset_key': 'key_1',
                    'bucket': 'bucket_1'
                }
            },
            'ModelMetrics': {
                'TrainingRun': [
                    {
                        'label': 'best',
                        'subset': 'valid',
                        'PredictionResults': {
                            'r2_score': 50
                        }
                    }
                ],
                'PredictionRuns': []
            }
        }
        check(self.client_wrapper, filter_dict, 'get', 'model', 200,
              match_list=[expected_match])

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
    metrics_4 = {
        'model_uuid': 'uuid_1',
        'ModelMetrics': {
            'PredictionRuns': {
                'PredictionResults': {
                    'r2_score': 3,
                    'num_compounds': 5
                }
            }
        }
    }
    check(client_wrapper, metrics_4, 'save', 'metrics', 200)

def save_dicts_2(client_wrapper):
    # Save two metadata dicts.
    metadata_1 = {
        'time_built': '2019-02-14',
        'model_uuid': 'uuid_1',
        'ModelMetadata': {
            'TrainingDataset': {
                'dataset_key': 'key_1',
                'bucket': 'bucket_1'
            }
        }
    }
    check(client_wrapper, metadata_1, 'save', 'metadata', 200)
    metadata_2 = {
        'time_built': '2019-02-14',
        'model_uuid': 'uuid_2',
        'ModelMetadata': {
            'TrainingDataset': {
                'dataset_key': 'key_2',
                'bucket': 'bucket_2'
            }
        }
    }
    check(client_wrapper, metadata_2, 'save', 'metadata', 200)
    # Save six metrics dicts.
    metrics_1 = {
        'model_uuid': 'uuid_1',
        'ModelMetrics': {
            'TrainingRun': {
                'label': 'best',
                'subset': 'train',
                'PredictionResults': {
                    'r2_score': 100
                }
            }
        }
    }
    check(client_wrapper, metrics_1, 'save', 'metrics', 200)
    metrics_2 = {
        'model_uuid': 'uuid_1',
        'ModelMetrics': {
            'TrainingRun': {
                'label': 'best',
                'subset': 'valid',
                'PredictionResults': {
                    'r2_score': 50
                }
            }
        }
    }
    check(client_wrapper, metrics_2, 'save', 'metrics', 200)
    metrics_3 = {
        'model_uuid': 'uuid_1',
        'ModelMetrics': {
            'TrainingRun': {
                'label': 'best',
                'subset': 'valid',
                'PredictionResults': {
                    'r2_score': 20
                }
            }
        }
    }
    check(client_wrapper, metrics_3, 'save', 'metrics', 200)
    metrics_4 = {
        'model_uuid': 'uuid_2',
        'ModelMetrics': {
            'TrainingRun': {
                'label': 'best',
                'subset': 'train',
                'PredictionResults': {
                    'r2_score': 90
                }
            }
        }
    }
    check(client_wrapper, metrics_4, 'save', 'metrics', 200)
    metrics_5 = {
        'model_uuid': 'uuid_2',
        'ModelMetrics': {
            'TrainingRun': {
                'label': 'best',
                'subset': 'valid',
                'PredictionResults': {
                    'r2_score': 70
                }
            }
        }
    }
    check(client_wrapper, metrics_5, 'save', 'metrics', 200)
    metrics_6 = {
        'model_uuid': 'uuid_2',
        'ModelMetrics': {
            'TrainingRun': {
                'label': 'best',
                'subset': 'valid',
                'PredictionResults': {
                    'r2_score': 10
                }
            }
        }
    }
    check(client_wrapper, metrics_6, 'save', 'metrics', 200)

# https://docs.pytest.org/en/latest/getting-started.html for more on test
# classes.
class TestModelsFailure(object):
    
    # https://docs.pytest.org/en/latest/xunit_setup.html for more on setup
    # methods.
    def setup_method(self):
        setup(self)

    def teardown_method(self):
        teardown(self)

    def test_models_failure_internal_error(self):
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
        check(self.client_wrapper, filter_dict, 'get', 'model', 500,
              expected_errors=errors)
