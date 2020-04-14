from utilities import *

# Run with `python3 -m pytest -s` in the directory this file is in.
# Run a specific test with `python3 -m pytest -s test_mlmt_client_port.py -k
# <test_name>`
# Also test using Swagger UI or by calling these functions from an IPython
# notebook.


# https://docs.pytest.org/en/latest/getting-started.html for more on test
# classes.
class TestPortSuccess(object):
    # https://docs.pytest.org/en/latest/xunit_setup.html for more on setup
    # methods.
    def setup_method(self):
        setup(self)

    def teardown_method(self):
        teardown(self)
        
    def test_port_success_already_formatted(self):
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
        check(self.client_wrapper, metadata_1, 'save', 'metadata', 200)
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
        check(self.client_wrapper, metadata_2, 'save', 'metadata', 200)
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
        check(self.client_wrapper, metrics_1, 'save', 'metrics', 200)
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
        check(self.client_wrapper, metrics_2, 'save', 'metrics', 200)
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
        check(self.client_wrapper, metrics_3, 'save', 'metrics', 200)
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
        check(self.client_wrapper, metrics_4, 'save', 'metrics', 200)
        output = self.client_wrapper.port_metrics(
            filter_dict={'collection_name' : COLLECTION_NAME})
        assert output['status'] == '200 OK'
        
    # Cannot test success on obsolete format, since there are no API functions
    # for saving with the obsolete format now.
    # That test is in the ml_services repo.
        

# https://docs.pytest.org/en/latest/getting-started.html for more on test
# classes.
class TestPortFailure(object):
    # https://docs.pytest.org/en/latest/xunit_setup.html for more on setup
    # methods.
    def setup_method(self):
        setup(self)

    def teardown_method(self):
        teardown(self)
        
    def test_port_failure_missing_collection_name(self):
        output = self.client_wrapper.port_metrics(
            filter_dict={})
        assert (output['status'], output['errors']) == (
            '400 Bad Request',
            'Status400Exception: collection_name is unspecified. keys=[]')

    def test_port_failure_internal_error(self):
        filter_dict = {
            'test_internal_error': True,
            'collection_name': COLLECTION_NAME
        }
        output = self.client_wrapper.port_metrics(
            filter_dict=filter_dict)
        assert (output['status'], output['errors']) == (
            '500 Internal Server Error',
            'Status500Exception: Testing internal error.')
