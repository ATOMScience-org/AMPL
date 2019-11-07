from utilities import *

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparent_dir = os.path.dirname(parentdir)
sys.path.insert(0,grandparent_dir)
import pipeline.mlmt_client_wrapper as mlmt_client_wrapper

# Run with `python3 -m pytest -s` in the directory this file is in.
# Run a specific test with `python3 -m pytest -s test_mlmt_client_
# collections.py -k <test_name>`
# Also test using Swagger UI or by calling these functions from an IPython
# notebook.

# https://docs.pytest.org/en/latest/getting-started.html for more on test
# classes.
class TestCollectionsSuccess(object):
    # https://docs.pytest.org/en/latest/xunit_setup.html for more on setup
    # methods.
    def setup_method(self):
        setup(self, collection_names=[
            COLLECTION_NAME, 'test_collection_a', 'test_collection_a_metrics',
            'test_collection_b', 'test_collection_b_metrics', 'test_z',
            'test_z_metrics'])
            
    def teardown_method(self):
        teardown(self, collection_names=[
            COLLECTION_NAME, 'test_collection_a', 'test_collection_a_metrics',
            'test_collection_b', 'test_collection_b_metrics', 'test_z',
            'test_z_metrics'])

    def test_collection_manipulation_success(self):
        create_dicts(self.client_wrapper)
        # Get all collection names.
        check_collection_names(self, {}, [
            'test_collection_a', 'test_collection_a_metrics',
            'test_collection_b', 'test_collection_b_metrics',
            'test_z', 'test_z_metrics'])
        # Get specific collection names.
        check_collection_names(self, {'substring' : 'collection'}, [
            'test_collection_a', 'test_collection_a_metrics',
            'test_collection_b', 'test_collection_b_metrics'])
        # See the contents of a metadata collection.
        expected = [
            {
                '_id': 'dummy',
                'model_uuid': 'uuid_1',
                'ModelMetadata': {},
                'ModelMetrics': {
                    'TrainingRun': [
                        {
                            'PredictionResults': {
                                'r2_score': 3,
                                'num_compounds': 5
                            }
                        }
                    ]
                }
            }
        ]
        check_collection(self.client_wrapper, {
            'collection_name': 'test_collection_a'}, expected)
        # See the contents of a metrics collection.
        expected = [
            {
                '_id': 'dummy',
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
        ]
        check_collection(self.client_wrapper, {
            'collection_name': 'test_collection_a_metrics'}, expected,
                         is_metrics=True)
        # Delete items by model_uuid
        check_delete_from_collection(self.client_wrapper, {
            'collection_name': 'test_z', 'model_uuids': ['uuid_1']})
        check_collection(self.client_wrapper, {'collection_name': 'test_z'}, [])
        check_collection(self.client_wrapper, {
            'collection_name': 'test_z_metrics'}, [])
        # Delete item by model_uuid, keeping metrics
        check_delete_from_collection(self.client_wrapper, {
            'collection_name' : 'test_collection_a', 'model_uuids': ['uuid_1'],
            'keep_metrics': True})
        check_collection(self.client_wrapper, {
            'collection_name': 'test_collection_a'}, [])
        expected = [
            {
                '_id': 'dummy',
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
        ]
        check_collection(self.client_wrapper, {
            'collection_name': 'test_collection_a_metrics'}, expected)
        # Delete collection.
        check_delete_collection(self.client_wrapper, {
            'collection_names': ['test_collection_a']})
        check_collection_names(self, {}, [
            'test_collection_b', 'test_collection_b_metrics', 'test_z',
            'test_z_metrics'])
        # Delete collection, keeping metrics
        filter_dict={
            'collection_names': ['test_collection_b'],
            'keep_metrics': True
        }
        check_delete_collection(self.client_wrapper, filter_dict)
        check_collection_names(self, {}, ['test_collection_b_metrics',
                                          'test_z', 'test_z_metrics'])
        
    def test_update_metadata_success(self):
        create_dicts_2(self.client_wrapper)
        updates_dict = {
            'collection_name': COLLECTION_NAME,
            'model_uuid': 'uuid_1',
            'ModelMetadata.ModelSpecific.model_type': 'RF'
        }
        check_update_metadata(self.client_wrapper, updates_dict)
        output = list(self.client_wrapper.get_metadata_generator(
            filter_dict=updates_dict))
        assert len(output) == 1
        original_dict = {
            'collection_name': COLLECTION_NAME,
            'model_uuid': 'uuid_1',
            'ModelMetadata.ModelSpecific.model_type': 'NN'
        }
        output = list(self.client_wrapper.get_metadata_generator(
            filter_dict=original_dict))
        assert len(output) == 0
        
    def test_get_collection_success(self):
        create_dicts_3(self.client_wrapper)
        expected = [
            {
                '_id': 'dummy',
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
                            'num': 1,
                            'PredictionResults': {
                                'r2_score': 3,
                                'num_compounds': 5
                            }
                        },
                        {
                            'num': 2,
                            'PredictionResults': {
                                'r2_score': 14,
                                'num_compounds': 7
                            }
                        }
                    ]
                }
            },
            {
                '_id': 'dummy',
                'time_built': '2018-11-06',
                'model_uuid': 'uuid_2',
                'ModelMetadata': {
                    'ModelSpecific': {
                        'model_type': 'RF',
                        'num_tasks': 3,
                        'uncertainty': False
                    }
                },
                'ModelMetrics': {
                    'TrainingRun': [
                        {
                            'num': 3,
                            'PredictionResults': {
                                'r2_score': 3,
                                'num_compounds': 5
                            }
                        }
                    ]
                }
            }
        ]
        check_collection(self.client_wrapper, {
            'collection_name': COLLECTION_NAME}, expected,
                         sort_function=lambda d: d['model_uuid'])
        expected = [
            {
                '_id': 'dummy',
                'model_uuid': 'uuid_1',
                'ModelMetrics': {
                    'TrainingRun': {
                        'num': 1,
                        'PredictionResults': {
                            'r2_score': 3,
                            'num_compounds': 5
                        }
                    }
                }
            },
            {
                '_id': 'dummy',
                'model_uuid': 'uuid_1',
                'ModelMetrics': {
                    'TrainingRun': {
                        'num': 2,
                        'PredictionResults': {
                            'r2_score': 14,
                            'num_compounds': 7
                        }
                    }
                }
            },
            {
                '_id': 'dummy',
                'model_uuid': 'uuid_2',
                'ModelMetrics': {
                    'TrainingRun': {
                        'num': 3,
                        'PredictionResults': {
                            'r2_score': 3,
                            'num_compounds': 5
                        }
                    }
                }
            },
            {
                '_id': 'dummy',
                'model_uuid': 'uuid_1',
                'ModelMetrics': {
                    'PredictionRuns': {
                        'num': 4,
                        'PredictionResults': {
                            'r2_score': 3,
                            'num_compounds': 5
                        }
                    }
                }
            }
        ]

        def sort(d):
            d1 = d['ModelMetrics']
            if 'TrainingRun' in d1.keys():
                return d1['TrainingRun']['num']
            elif 'PredictionRuns' in d1.keys():
                return d1['PredictionRuns']['num']
            else:
                raise Exception('Invalid Keys={keys}'.format(
                    keys=str(d1.keys())))
        check_collection(self.client_wrapper, {
            'collection_name': COLLECTION_NAME + '_metrics'}, expected,
                         sort_function=sort)


# https://docs.pytest.org/en/latest/getting-started.html for more on test
# classes.
class TestCollectionsFailure(object):
    # https://docs.pytest.org/en/latest/xunit_setup.html for more on setup
    # methods.
    def setup_method(self):
        setup(self, collection_names=[
            COLLECTION_NAME, 'test_collection_a', 'test_collection_a_metrics',
            'test_collection_b', 'test_collection_b_metrics', 'test_z',
            'test_z_metrics'])
            
    def teardown_method(self):
        teardown(self, collection_names=[
            COLLECTION_NAME, 'test_collection_a', 'test_collection_a_metrics',
            'test_collection_b', 'test_collection_b_metrics', 'test_z',
            'test_z_metrics'])
        
    def test_collection_manipulation_failure_internal_error(self):
        create_dicts(self.client_wrapper)
        # Test update_metadata.
        output = self.client_wrapper.update_metadata(
            updates_dict={'collection_name': COLLECTION_NAME,
                          'test_internal_error': True})
        assert (output['status'], output['errors']) == (
            '500 Internal Server Error',
            'Status500Exception: Testing internal error.')
        # Test get_collection_names.
        output = self.client_wrapper.get_collection_names(
            filter_dict={'test_internal_error': True})
        assert (output['status'], output['errors']) == (
            '500 Internal Server Error',
            'Status500Exception: Testing internal error.')
        # Test get_collection.
        try:
            output = list(self.client_wrapper.get_collection_generator(
                filter_dict={
                    'collection_name': COLLECTION_NAME,
                    'test_internal_error': True
                }))
        except mlmt_client_wrapper.MongoQueryException as e:
            actual = str(e)
            expected = ('Failed to query ids. Status:'
                        ' 500 Internal Server Error, Errors:'
                        ' Status500Exception: Testing internal error.')
            assert actual == expected
        # Test delete_from_collection_internal.
        output = self.client_wrapper.delete_from_collection(
                filter_dict={'test_internal_error': True})
        assert (output['status'], output['errors']) == (
            '500 Internal Server Error',
            'Status500Exception: Testing internal error.')
        # Test delete_collections_internal.
        output = self.client_wrapper.delete_collections(
            filter_dict={'test_internal_error': True})
        assert (output['status'], output['errors']) == (
            '500 Internal Server Error',
            'Status500Exception: Testing internal error.')
        
    def test_update_metadata_failure_missing_uuid(self):
        create_dicts_2(self.client_wrapper)
        updates_dict = {
            'collection_name': COLLECTION_NAME
        }
        output = self.client_wrapper.update_metadata(
            updates_dict=updates_dict)
        assert (output['status'], output['errors']) == (
            '400 Bad Request',
            ('Status400Exception: updates_dict does not contain'
             ' the key model_uuid.'))
        
    def test_update_metadata_failure_invalid_uuid(self):
        create_dicts_2(self.client_wrapper)
        updates_dict = {
            'collection_name': COLLECTION_NAME,
            'model_uuid': 'uuid_c'
        }
        output = self.client_wrapper.update_metadata(
            updates_dict=updates_dict)
        assert (output['status'], output['errors']) == (
            '400 Bad Request',
            'Status400Exception: No model has model_uuid=uuid_c.')
        
    def test_get_collection_names_failure(self):
        create_dicts(self.client_wrapper)
        output = self.client_wrapper.get_collection_names(
            filter_dict={'x': 5})
        assert (output['status'], output['errors']) == (
            '400 Bad Request', "Status400Exception: Invalid keys=['x'].")
        
    def test_get_collection_failure_invalid_keys(self):
        create_dicts(self.client_wrapper)
        try:
            output = list(self.client_wrapper.get_collection_generator(
                filter_dict={'x': 5, 'collection_name': 'test_z'}))
        except mlmt_client_wrapper.MongoQueryException as e:
            actual = str(e)
            expected = "Invalid keys=['x']"
            assert actual == expected
        
    def test_get_collection_failure_no_collection(self):
        create_dicts(self.client_wrapper)
        try:
            output = list(
                self.client_wrapper.get_collection_generator(filter_dict={}))
        except mlmt_client_wrapper.MongoQueryException as e:
            actual = str(e)
            expected = 'collection_name is unspecified.'
            assert actual == expected
        
    def test_delete_from_collection_failure_invalid_key_combination(self):
        create_dicts(self.client_wrapper)
        output = self.client_wrapper.delete_from_collection(
                filter_dict={'ids' : [], 'model_uuids': [],
                             'collection_name' : 'test_z'})
        assert (output['status'], output['errors']) == (
            '400 Bad Request',
            "Status400Exception: Cannot specify both ids and model_uuids.")
        
    def test_delete_from_collection_failure_invalid_keys(self):
        create_dicts(self.client_wrapper)
        output = self.client_wrapper.delete_from_collection(
                filter_dict={'x': 5, 'collection_name': 'test_z'})
        assert (output['status'], output['errors']) == (
            '400 Bad Request', "Status400Exception: Invalid keys=['x'].")
        
    def test_delete_from_collection_failure_no_collection(self):
        create_dicts(self.client_wrapper)
        output = self.client_wrapper.delete_from_collection(
                filter_dict={})
        assert (output['status'], output['errors']) == (
            '400 Bad Request',
            'Status400Exception: collection_name is unspecified.')
        
    def test_delete_collections_failure_no_collections(self):
        create_dicts(self.client_wrapper)
        output = self.client_wrapper.delete_collections(
            filter_dict={})
        assert (output['status'], output['errors']) == (
            '400 Bad Request',
            "Status400Exception: collection_names not specified.")
        
    def test_delete_collections_failure_invalid_keys(self):
        create_dicts(self.client_wrapper)
        output = self.client_wrapper.delete_collections(
            filter_dict={'collection_names' : ['test_z'], 'x': 5})
        assert (output['status'], output['errors']) == (
            '400 Bad Request', "Status400Exception: Invalid keys=['x'].")
        
    def test_delete_collections_failure_disallow_deletion(self):
        create_dicts(self.client_wrapper)
        output = self.client_wrapper.delete_collections(
            filter_dict={'collection_names' : ['z']})
        assert (output['status'], output['errors']) == (
            '400 Bad Request',
            ('Status400Exception: collection_name=z does not'
             ' start with "test_" and therefore cannot be deleted.'))


def check_collection_names(test_class, filter_dict, expected_collections,
                           log=False):
    output = test_class.client_wrapper.get_collection_names(
            filter_dict=filter_dict)
    if log and (output['status'] != '200 OK'):
        print(output['errors'])
        print(output['trace'])
    assert output['status'] == '200 OK'
    substring = filter_dict.pop('substring', None)
    if substring is None:
        expected = sorted(expected_collections + test_class.initial_collections)
    else:
        initial_collection_matches = []
        for c in test_class.initial_collections:
            if substring in c:
                initial_collection_matches.append(c)
        expected = sorted(expected_collections + initial_collection_matches)
    assert output['matching_collection_names'] == expected


def check_collection(client_wrapper, filter_dict, expected, is_metrics=False,
                     sort_function=None, log=False):
    actual = list(client_wrapper.get_collection_generator(
        filter_dict=filter_dict, log=log))
    if sort_function is not None:
        # https://stackoverflow.com/questions/72899/how-do-i-sort-a-list-of-dictionaries-by-a-value-of-the-dictionary
        actual = sorted(actual, key=sort_function) 
    for d in actual:
        d['_id'] = 'dummy'
    if log:
        for d in actual:
            print(d)
        print('\n')
        for d in expected:
            print(d)
    assert actual == expected            


def check_collection_length(client_wrapper, filter_dict, expected_num, log=False):
    output = list(client_wrapper.get_collection_generator(
        filter_dict=filter_dict))
    assert len(output) == expected_num


def check_delete_from_collection(client_wrapper, filter_dict, log=False):
    output = client_wrapper.delete_from_collection(
        filter_dict=filter_dict)
    if log and (output['status'] != '200 OK'):
        print(output['errors'])
        print(output['trace'])
    assert output['status'] == '200 OK'


def check_delete_collection(client_wrapper, filter_dict, log=False):
    output = client_wrapper.delete_collections(
        filter_dict=filter_dict)
    if log and (output['status'] != '200 OK'):
        print(output['errors'])
        print(output['trace'])
    assert output['status'] == '200 OK'


def check_update_metadata(client_wrapper, updates_dict, log=False):
    output = client_wrapper.update_metadata(
        updates_dict=updates_dict)
    if log and (output['status'] != '200 OK'):
        print(output['errors'])
        print(output['trace'])
    assert output['status'] == '200 OK'


def create_dicts(client_wrapper):
    # Create metadata dicts.
    d = {
        'model_uuid' : 'uuid_1',
        'ModelMetadata' : {}
    }
    check(client_wrapper, d, 'save', 'metadata', 200,
          collection_name='test_collection_a')
    d = {
        'model_uuid' : 'uuid_1',
        'ModelMetadata' : {}
    }
    check(client_wrapper, d, 'save', 'metadata', 200,
          collection_name='test_collection_b')
    d = {
        'model_uuid' : 'uuid_1',
        'ModelMetadata' : {}
    }
    check(client_wrapper, d, 'save', 'metadata', 200,
          collection_name='test_z')
    # Create metrics dicts.
    d = {
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
    check(client_wrapper, d, 'save', 'metrics', 200,
          collection_name='test_collection_a')
    d = {
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
    check(client_wrapper, d, 'save', 'metrics', 200,
          collection_name='test_collection_b')
    d = {
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
    check(client_wrapper, d, 'save', 'metrics', 200, collection_name='test_z')
    
def create_dicts_2(client_wrapper):
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
    # The key 'num' is for easy sorting in tests.
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


def create_dicts_3(client_wrapper):
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
    # The key 'num' is for easy sorting in tests.
    metrics_1 = {
        'model_uuid': 'uuid_1',
        'ModelMetrics': {
            'TrainingRun': {
                'num': 1,
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
                'num': 2,
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
                'num': 3,
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
                'num': 4,
                'PredictionResults': {
                    'r2_score': 3,
                    'num_compounds': 5
                }
            }
        }
    }
    check(client_wrapper, metrics_4, 'save', 'metrics', 200)
