import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparent_dir = os.path.dirname(parentdir)
ggparent_dir = os.path.dirname(grandparent_dir)
gggparent_dir = os.path.dirname(ggparent_dir)
sys.path.insert(0,gggparent_dir)
import pipeline.mlmt_client_wrapper as mlmt_client_wrapper

USE_PRODUCTION_SERVER = True
COLLECTION_NAME = 'test_mlmt_client'


def check(client_wrapper, d, action, dict_type, expected_status,
          collection_name=COLLECTION_NAME, expected_errors=None, match_list=[],
          non_match_list=[], asserting_all_matches=False, add_indexes=False,
          log=False):
    if collection_name is not None:
        d['collection_name'] = collection_name
    else:
        raise Exception(
            ('collection_name has been explicitly set to None.'
             ' Default={default}.').format(default=COLLECTION_NAME))
    # Determine expected status.
    if expected_status == 200:
        expected_status = '200 OK'
    elif expected_status == 400:
        expected_status = '400 Bad Request'
    elif expected_status == 500:
        expected_status = '500 Internal Server Error'
    else:
        raise Exception('Invalid expected status: {expected_status}'.format(
            expected_status=expected_status))
    # Run command.
    if action == 'save':
        if not add_indexes:
            # To make tests run faster, skip indexing.
            d['add_indexes'] = False
        if dict_type == 'metadata':
            try:
                client_wrapper.save_metadata(model_metadata_dict=d, log=log)
            except mlmt_client_wrapper.MongoInsertionException as e:
                actual = str(e)
                expected = (
                    'Failed to insert model metadata into database.'
                    ' Status: {status}, Errors: {errors}').format(
                    status=expected_status,
                    errors=expected_errors
                )
                if log:
                    print('actual={actual}'.format(actual=actual))
                    print('expected={expected}'.format(expected=expected))
                assert actual == expected
        elif dict_type == 'metrics':
            try:
                client_wrapper.save_metrics(model_metrics_dict=d, log=log)
            except mlmt_client_wrapper.MongoInsertionException as e:
                actual = str(e)
                expected = (
                    'Failed to insert model metrics into database.'
                    ' Status: {status}, Errors: {errors}').format(
                    status=expected_status,
                    errors=expected_errors
                )
                if log:
                    print('actual={actual}'.format(actual=actual))
                    print('expected={expected}'.format(expected=expected))
                assert actual == expected
        else:
            raise Exception('Invalid dict_type: {dict_type}'.format(
                dict_type=dict_type))
        if not add_indexes:
            d.pop('add_indexes')
    elif action == 'get':
        try:
            if dict_type == 'metadata':
                matching_dicts = list(client_wrapper.get_metadata_generator(
                    filter_dict=d, log=log))
            elif dict_type == 'metrics':
                matching_dicts = list(client_wrapper.get_metrics_generator(
                    filter_dict=d, log=log))
            elif dict_type == 'model':
                matching_dicts = list(client_wrapper.get_models_generator(
                    filter_dict=d, log=log))
            else:
                raise Exception('Invalid dict_type: {dict_type}'.format(
                    dict_type=dict_type))
            if log:
                print('Listing actual matching_dicts:')
            for matching_dict in matching_dicts:
                matching_dict.pop('_id', None)
                if log:
                    print(matching_dict)
            if log:
                print('Listing expected match_list:')
            for match in match_list:
                match.pop('_id', None)
                match.pop('collection_name', None)
                if log:
                    print(match)
                    print('Asserting the above is in actual matching_dicts.')
                assert match in matching_dicts
            if asserting_all_matches:
                # If we are asserting all matches that then the following should
                # be true.
                assert len(matching_dicts) == len(match_list)
            if log:
                print('Listing expected non_match_list:')
            for non_match in non_match_list:
                non_match.pop('_id', None)
                non_match.pop('collection_name', None)
                if log:
                    print(non_match)
                    print(('Asserting the above is not in actual'
                           ' matching_dicts.'))
                assert non_match not in matching_dicts
            if match_list == []:
                assert matching_dicts == []
        except mlmt_client_wrapper.MongoQueryException as e:
            actual = str(e)
            expected = ('Failed to query ids.'
                        ' Status: {status}, Errors: {errors}').format(
                status=expected_status,
                errors=expected_errors
            )
            if log:
                print('actual={actual}'.format(actual=actual))
                print('expected={expected}'.format(expected=expected))
            assert actual == expected
    else:
        raise Exception('Invalid action: {action}'.format(action=action))


def setup(test_class, collection_names=[COLLECTION_NAME], log=False):
    # Set up mlmt_client.
    test_class.client_wrapper = mlmt_client_wrapper.MLMTClientWrapper()
    test_class.client_wrapper.instantiate_mlmt_client(
        use_production_server=USE_PRODUCTION_SERVER)
    mlmt_client = test_class.client_wrapper.mlmt_client
    # Check mlmt_client is configured properly.
    assert mlmt_client is not None
    assert dir(mlmt_client.port) == ['port_model_metrics']
    assert dir(mlmt_client.collection_manipulation) == [
        'delete_collections', 'delete_from_collection',
        'get_collection_names', 'update_model_metadata']
    assert dir(mlmt_client.ids) == ['get_by_id', 'get_ids']
    assert dir(mlmt_client.model_metadata) == ['save_model_metadata']
    assert dir(mlmt_client.model_metrics) == ['save_model_metrics']
    # Delete the test collection if it exists, so we can start fresh.
    # This will also delete COLLECTION_NAME + '_metrics'.
    output = test_class.client_wrapper.delete_collections(
        filter_dict={'collection_names': collection_names})
    if log and (output['status'] != '200 OK'):
        print(output['errors'])
        print(output['trace'])
    assert output['status'] == '200 OK'
    # Get the current number of collections.
    output = \
        test_class.client_wrapper.get_collection_names(
            filter_dict={})
    if log and (output['status'] != '200 OK'):
        print(output['errors'])
        print(output['trace'])
    assert output['status'] == '200 OK'
    test_class.num_initial_collections = len(
        output['matching_collection_names'])
    test_class.initial_collections = output['matching_collection_names']


def teardown(test_class, collection_names=[COLLECTION_NAME], log=False):
    # Delete the collection.
    # This will also delete COLLECTION_NAME + '_metrics'.
    output = test_class.client_wrapper.delete_collections(
        filter_dict={'collection_names' : collection_names})
    if log and (output['status'] != '200 OK'):
        print(output['errors'])
        print(output['trace'])
    assert output['status'] == '200 OK'
    # Check that the number of collections has been restored to its previous
    # value.
    output = test_class.client_wrapper.get_collection_names(
        filter_dict={})
    if log and (output['status'] != '200 OK'):
        print(output['errors'])
        print(output['trace'])
    assert output['status'] == '200 OK'
    assert len(output['matching_collection_names']) == \
           test_class.num_initial_collections
    assert output['matching_collection_names'] == \
           test_class.initial_collections
