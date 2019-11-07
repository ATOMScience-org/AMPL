import os
import sys

from atomsci.ddm.utils import datastore_functions as dsf
import copy


class MongoInsertionException(Exception):
    pass


class MongoQueryException(Exception):
    pass


class MLMTClientWrapper(object):
    def __init__(self, mlmt_client=None, ds_client=None):
        """Create the wrapper.

        Args:
            mlmt_client: The mlmt_client that should be wrapped. Default None.
            
            ds_client: The ds_client. Default None.

        """
        self.mlmt_client = mlmt_client
        self.ds_client = ds_client

    def instantiate_mlmt_client(self, use_production_server=True):
        """Instantiate the mlmt_client.

        Args:
            use_production_server (bool): True if production server should be
            used. False if local server should be used. Default True. Local
            server should only be used for testing.
            
        """
        
        # =====================================================
        # Set up machine learning model tracker (mlmt) client.
        # =====================================================
        # Toggle True/False to use production server or the forsyth2 personal
        # server.
        # The former should almost always be used, unless testing with code only
        # running on the latter.
        self.ds_client, self.mlmt_client = dsf.initialize_model_tracker(use_production_server,self.ds_client) 

    def save_metadata(self, model_metadata_dict, log=False):
        """Save model metadata to MongoDB.

        Args:
            model_metadata_dict (dict): The dictionary of model metadata.
            Must be of the following form:

                {

                    time_built: Time the model was built,

                    model_uuid: Unique identifier for this model,

                    ModelMetadata: {...} # The metadata to save.

                    # Special Keys:

                    collection_name (str): The Mongo collection to be used.

                    add_indexes (bool, optional): If True, then add indexes.
                    Default False.

                    test_internal_error (bool, optional): If True, trigger a
                    500 error. For testing purposes only. Default False.

                }

            log (bool): True if logs should print. False otherwise.
            Default False.



        Returns:
            dict: Output dictionary of the form:

                {

                    'status': The status code of the save operation,
                    # one of (200 OK, 400 Bad Request, or 500 Internal Server
                    Error).

                    # These two keys may or may not be present:

                    'errors': Errors encountered,

                    'trace' : Stack Trace of fatal error

                }
                
        """
        output = self.mlmt_client.model_metadata.save_model_metadata(
            model_metadata_dict=model_metadata_dict).result()
        if output is None:
            raise MongoInsertionException(
                'Failed to insert model_metadata into database.'
                ' output not returned.')
        status = output['status']
        if status != '200 OK':
            if log:
                print('save_metadata:\n' + output['trace'])
            raise MongoInsertionException(
                ('Failed to insert model metadata into database.'
                 ' Status: {status},' 
                 ' Errors: {errors}').format(
                    status=status,
                    errors=output['errors']
                )
            )

    def save_metrics(self, model_metrics_dict, log=False):
        """Save model metrics to MongoDB.

        Args:
            model_metrics_dict (dict): The dictionary of model metrics.
            Must be of the following form:

                {

                  'model_uuid' : The model_uuid of the model run to produce
                  these metrics

                  'ModelMetrics' : {

                    'TrainingRun' : [

                      {

                        'label': 'best',

                        'subset' : 'train',

                        'PredictionResults' : {...}

                      },

                     ...

                     ], # TrainingRun can also just be a single {...}

                   'PredictionRuns': [{...}, ...] # or {...}

                  },

                  # Special Keys:

                  collection_name (str): The Mongo collection to be used.
                  Keep in mind that this function will automatically add
                  '_metrics' to the end of the collection name.

                  add_indexes (bool, optional): If True, then add indexes.
                  Default False.

                  test_internal_error (bool, optional): If True, trigger a 500
                  error. For testing purposes only. Default False.

                }

            log (bool): True if logs should print. False otherwise.
            Default False.

        Returns:
            dict: Output dictionary of the form:

                {

                    'status': The status code of the save operation,
                    # one of (200 OK, 400 Bad Request, or 500 Internal Server
                    Error).

                    # These two keys may or may not be present:

                    'errors': Errors encountered,

                    'trace' : Stack Trace of fatal error

                }
                
        """
        output = self.mlmt_client.model_metrics.save_model_metrics(
            model_metrics_dict=model_metrics_dict).result()
        if output is None:
            raise MongoInsertionException(
                ('Failed to insert model_metrics into database.'
                 ' output not returned.')
            )
        status = output['status']
        if status != '200 OK':
            if log:
                print('save_metrics:\n' + output['trace'])
            raise MongoInsertionException(
                ('Failed to insert model metrics into database.'
                 ' Status: {status}, Errors: {errors}').format(
                    status=status,
                    errors=output['errors']
                )
            )

    def get_models_generator(self, filter_dict, log=False):
        """Get specific objects in a collection.

        ``get_model_metadata_generator`` will return only the metadata portion
        of matching dicts.

        ``get_model_metrics_generator`` will return only the metrics portion of
        matching dicts.

        ``get_collection_generator`` will return the entire collectiion.
        Furthermore, this only returns the metadata collection or the metrics
        collection.

        This function will return specific metadata (along with appropriate
        metrics inline) from the collection.

        Args:
            filter_dict (dict): The dictionary to filter results on.
            Hierarchy is represented with dots, as follows:

                {

                    'ModelMetadata.ModelParameters.model_type' : 'NN',

                    'ModelMetrics.TrainingRun.PredictionResults.r2_score' : 0.5,

                    # Special Keys:

                    collection_name (str): The Mongo collection to be used.

                    pop_id (bool, optional): If True, pop the '_id' field from
                    the return dicts. Default True.

                    test_internal_error (bool, optional): If True, trigger a
                    500 error. For testing purposes only. Default False.

                }

            log (bool): True if logs should print. False otherwise.
            Default False.

        Returns:
            generator of dictionaries of the form:
                {

                    'status': The status code of the get operation,
                    # one of (200 OK, 400 Bad Request, or 500 Internal Server
                    Error)

                    # These three keys may or may not be present:

                    'item': A matching model (metadata plus matching metrics)

                    'errors': Errors encountered,

                    'trace' : Stack Trace of fatal error
                }
                
        """
        metadata_filter_dict = copy.deepcopy(filter_dict)
        keys = list(metadata_filter_dict.keys())
        for key in keys:
            if isinstance(metadata_filter_dict[key], list):
                if metadata_filter_dict[key][0] in ['min', 'max']:
                    if key.startswith('ModelMetrics.'):
                        # If we apply a metrics min/max on metadata and query
                        # for the metrics min/max given the acceptable
                        # metadata, then only the global maximum will be
                        # returned.
                        metadata_filter_dict.pop(key)
        metadata_gen = self.get_metadata_generator(metadata_filter_dict, log=log)
        for metadata_item in metadata_gen:
            # Look for metrics under this model_uuid.
            # Without this deepcopy, a new model will be returned for every metric.
            metrics_filter_dict = copy.deepcopy(filter_dict)
            metrics_filter_dict['model_uuid'] = metadata_item['model_uuid']
            if log: 
                print('get_models_generator: model_uuid={uuid}.'.format(
                    uuid=metadata_item['model_uuid']))
            # metadata already has TrainingRun in its dict, BUT
            # it has all of them, not just the matching ones.
            # Need to add MATCHING TrainingRun and PredictionRuns.
            if 'ModelMetrics' not in metadata_item.keys():
                metadata_item['ModelMetrics'] = {}
            metadata_item['ModelMetrics']['TrainingRun'] = []
            metadata_item['ModelMetrics']['PredictionRuns'] = []
            associated_metrics_gen = self.get_metrics_generator(metrics_filter_dict,
                                                                log=log)
            for associated_metrics in associated_metrics_gen:
                if log:
                    print(('get_models_generator:'
                           ' associated_metrics={metrics}.').format(
                        metrics=associated_metrics))
                if 'TrainingRun' in associated_metrics['ModelMetrics']:
                    metadata_item['ModelMetrics']['TrainingRun'].append(
                        associated_metrics['ModelMetrics']['TrainingRun'])
                if 'PredictionRuns' in associated_metrics['ModelMetrics']:
                    metadata_item['ModelMetrics']['PredictionRuns'].append(
                        associated_metrics['ModelMetrics']['PredictionRuns'])
            if (metadata_item['ModelMetrics']['TrainingRun'] == []) and\
                    (metadata_item['ModelMetrics']['PredictionRuns'] == []):
                # No matching metrics.
                # Still want to return matching metadata though.
                if log:
                    print('get_model_generator: no matching metrics.')
            # metadata_item now contains the corresponding metrics.
            yield metadata_item

    def get_full_metadata_generator(self, filter_dict, log=False):
        """Query model metadata (plus its TrainingRun metrics) from MongoDB.

        Args:
            filter_dict (dict): The dictionary to filter macthing model metadata dicts on.
            Hierarchy is represented with dots, as follows:

                {

                    'ModelMetadata.ModelParameters.model_type' : 'NN'

                    # Special Keys:

                    collection_name (str): The Mongo collection to be used.

                    pop_id (bool, optional): If True, pop the '_id' field from
                    the returned dicts. Default True.

                    test_internal_error (bool, optional): If True, trigger a
                    500 error. For testing purposes only. Default False.
                }

            log (bool): True if logs should print. False otherwise.
            Default False.

        Returns:
            generator of dictionaries of the form:

                {

                    'status': The status code of the get operation,
                    # one of (200 OK, 400 Bad Request, or 500 Internal Server
                    Error)

                    # These three keys may or may not be present:

                    'item': A matching metadata item,

                    'errors': Errors encountered,

                    'trace' : Stack Trace of fatal error
                }
                
        """
        filter_dict['return_type'] = 'metadata_training_run'
        return get_generator(self.mlmt_client, filter_dict, log=log)

    def get_metadata_generator(self, filter_dict, log=False):
        """Query model metadata from MongoDB.

        Args:
            filter_dict (dict): The dictionary to filter macthing model
            metadata dicts on. Hierarchy is represented with dots, as follows:

                {

                    'ModelMetadata.ModelParameters.model_type' : 'NN'

                    # Special Keys:

                    collection_name (str): The Mongo collection to be used.

                    pop_id (bool, optional): If True, pop the '_id' field from
                    the returned dicts. Default True.

                    test_internal_error (bool, optional): If True, trigger a
                    500 error. For testing purposes only. Default False.
                }

            log (bool): True if logs should print. False otherwise.
            Default False.

        Returns:
            generator of dictionaries of the form:

                {

                    'status': The status code of the get operation,
                    # one of (200 OK, 400 Bad Request, or 500 Internal Server
                    Error)

                    # These three keys may or may not be present:

                    'item': A matching metadata item,

                    'errors': Errors encountered,

                    'trace' : Stack Trace of fatal error
                }
                
        """
        filter_dict['return_type'] = 'metadata'
        return get_generator(self.mlmt_client, filter_dict, log=log)

    def get_metrics_generator(self, filter_dict, log=False):
        """Query model metrics from MongoDB.

        Args:
            filter_dict (dict): The dictionary to filter macthing model
            metrics dicts on. Hierarchy is represented with dots, as follows:

                {

                    'ModelMetrics.TrainingRun.PredictionResults.r2_score' : 0.5,

                    # Special Keys:

                    collection_name (str): The Mongo collection to be used.

                    pop_id (bool, optional): If True, pop the '_id' field from
                    the return dicts. Default True.

                    test_internal_error (bool, optional): If True, trigger a
                    500 error. For testing purposes only. Default False.

                }

            log (bool): True if logs should print. False otherwise.
            Default False.

        Returns:
            generator of dictionaries of the form:

                {

                    'status': The status code of the get operation,
                    # one of (200 OK, 400 Bad Request, or 500 Internal Server Error)

                    # These three keys may or may not be present:

                    'item': A matching metrics item,

                    'errors': Errors encountered,

                    'trace' : Stack Trace of fatal error
                }
                
        """
        filter_dict['return_type'] = 'metrics'
        return get_generator(self.mlmt_client, filter_dict, log=log)

    def get_collection_generator(self, filter_dict, log=False):
        """Get all items in a collection.

        Args:
            filter_dict (dict): The dictionary to filter results on.
            Must be of the following form:

                {

                    'collection_name': The collection to return

                    # Special Key:

                    test_internal_error (bool, optional): If True, trigger a
                    500 error. For testing purposes only. Default False.

                }

            log (bool): True if logs should print. False otherwise.
            Default False.

        Returns:
            generator of dictionaries of the form:

                {

                    'status': The status code of the get operation,
                    # one of (200 OK, 400 Bad Request, or 500 Internal Server
                    Error)

                    # These three keys may or may not be present:

                    'item': An item in the collection (metadata if it's a main
                    collection, metrics if it's a metrics collection),

                    'errors': Errors encountered,

                    'trace' : Stack Trace of fatal error

                }
                
        """
        keys = list(filter_dict.keys())
        try:
            keys.remove('collection_name')
        except ValueError:
            raise MongoQueryException('collection_name is unspecified.')
        if (keys != []) and (keys != ['test_internal_error']):
            raise MongoQueryException('Invalid keys={keys}'.format(keys=keys))
        collection_name = filter_dict['collection_name']
        if '_metrics' in collection_name:
            filter_dict['return_type'] = 'metrics_document'
            filter_dict['collection_name'] = filter_dict['collection_name'][:-8]
        else:
            filter_dict['return_type'] = 'metadata_document'
        # Get all contents of collection_name.
        return get_generator(self.mlmt_client, filter_dict, log=log)

    def update_metadata(self, updates_dict):
        """Update model metadata in MongoDB.

        Args:
            updates_dict (dict): The dictionary of model metadata. Hierarchy is
            represented with dots, as follows:

                {

                    'model_uuid': model_uuid of the model to update,

                    'ModelMetadata.ModelSpecific.model_type': 'RF'
                    # The metadata to update,

                    # Special Keys:

                    collection_name (str): The Mongo collection to be used.

                    test_internal_error (bool, optional): If True, trigger a
                    500 error. For testing purposes only. Default False.

                }

        Returns:
            dict: Output dictionary of the form:

                {

                    'status': The status code of the update operation, one of
                    (200 OK, 400 Bad Request, or 500 Internal Server Error).

                    # These two keys may or may not be present:

                    'errors': Errors encountered,

                    'trace' : Stack Trace of fatal error

                }
                
        """
        return self.mlmt_client.collection_manipulation.update_model_metadata(
            updates_dict=updates_dict).result()

    def get_collection_names(self, filter_dict):
        """Get the names of all existing collections.

        Args:
            filter_dict (dict): The dictionary to filter results on.
            Must be of the following form:

                {

                    'substring': If a collection name includes this substring
                    then it will be returned. If excluded then all collection
                    names are returned.

                    # Special Key:
                    test_internal_error (bool, optional): If True, trigger a
                    500 error. For testing purposes only. Default False.

                }

        Returns:
            dict: Output dictionary of the form:

                {

                    'status': The status code of the get operation, one of
                    (200 OK, 400 Bad Request, or 500 Internal Server Error).

                    # These three keys may or may not be present:

                    'matching_collection_names': The list of matches,

                    'errors': Errors encountered,

                    'trace' : Stack Trace of fatal error

                }
                
        """
        return self.mlmt_client.collection_manipulation.get_collection_names(
            filter_dict=filter_dict).result()

    def delete_from_collection(self, filter_dict):
        """Delete specified items from a collection.

        Args:
            filter_dict (dict): The dictionary containing information on which
            items to delete. Must be one of the following forms:

                {

                    'collection_name' : The collection to delete from,

                    'ids': List of items to delete filtered by '_id'

                }

                or

                {

                    'collection_name' : The collection to delete from,

                    'keep_metrics' : True if assoicated metrics should NOT
                    also be deleted; If excluded then defaults to False,

                    'model_uuids': List of items to delete, filtered by
                    'model_uuid'

                }

            The former allows for deletion of specific items. The latter allows
            for bulk deletes of a model's metadata and all its metrics.
            The former is not fully supported yet - see
            https://lc.llnl.gov/jira/browse/ATOM-204.

            Special Key:

            test_internal_error (bool, optional): If True, trigger a 500 error.
            For testing purposes only. Default False.

        Returns:
            dict: Output dictionary of the form:

                {

                    'status': The status code of the delete operation, one of
                    (200 OK, 400 Bad Request, or 500 Internal Server Error).

                    # These two keys may or may not be present:

                    'errors': Errors encountered,

                    'trace' : Stack Trace of fatal error

                }
                
        """
        return self.mlmt_client.collection_manipulation.delete_from_collection(
            filter_dict=filter_dict).result()

    def delete_collections(self, filter_dict):
        """Delete specified collections.

        Args:
            filter_dict (dict): The dictionary containing names of collections
            to be deleted. Must be of the following form:

                {

                    'collection_names': List of collection names to be deleted.
                    If excluded then all collection names are returned.

                    'keep_metrics' : True if the behind-the-scenes collection
                    ``collection_name + '_metrics'`` should NOT also be
                    deleted. If excluded then defaults to False.

                    # Special Key:

                    test_internal_error (bool, optional): If True, trigger a
                    500 error. For testing purposes only. Default False.

                }

        Returns:
            dict: Output dictionary of the form:

                {

                    'status': The status code of the delete operation, one of
                    (200 OK, 400 Bad Request, or 500 Internal Server Error).

                    # These two keys may or may not be present:

                    'errors': Errors encountered,

                    'trace' : Stack Trace of fatal error

                }
                
        """
        return self.mlmt_client.collection_manipulation.delete_collections(
            filter_dict=filter_dict).result()

    def port_metrics(self, filter_dict):
        """Port data to latest database schema.

        Args:
            filter_dict (dict): The dictionary to filter results on.
            Must be of the following form:

                {

                    'collection_name': The collection to port data in.

                    # Special Key:

                    test_internal_error (bool, optional): If True, trigger a
                    500 error. For testing purposes only. Default False.
                }

        Returns:
            dict: Output dictionary of the form:

                {

                    'status': The status code of the get operation, one of
                    (200 OK, 400 Bad Request, or 500 Internal Server Error).

                    # These two keys may or may not be present:

                    'errors': Errors encountered,

                    'trace' : Stack Trace of fatal error

                }
                
        """
        return self.mlmt_client.port.port_model_metrics(
            filter_dict=filter_dict).result()


def get_generator(mlmt_client, filter_dict, log=False):
    skip = 0
    limit = 1
    keys = list(filter_dict.keys())
    if log:
        print('get_generator: filter_dict_1={filter_dict}.'.format(
            filter_dict=filter_dict))
    if 'collection_name' not in keys:
        raise MongoQueryException('collection_name is unspecified.')
    if 'return_type' not in keys:
        raise MongoQueryException('return_type is unspecified.')
    already_have_ids = False
    for key in keys:
        if key not in ['collection_name', 'return_type', '_id']:
            # We have a non-id key that we need to filter on.
            # Therefore, we still need to narrow down the ids we have.
            break
        elif key == '_id':
            id_value = filter_dict['_id']
            if log:
                print('get_generator: id_value={v}.'.format(v=id_value))
            if isinstance(id_value, str):
                ids = [id_value]
            elif isinstance(id_value, dict) and ('$in' in id_value.keys()):
                ids = id_value['$in']
            elif isinstance(id_value, list) and (id_value[0] == 'in'):
                ids = id_value[1]
            else:
                # If id_value is something else, say 'nin' then we don't
                # actually have the exact ids we want.
                continue
            # We have all the ids we need.
            already_have_ids = True
            if log:
                print('get_generator: already have ids.')
            break
    while True:
        if not already_have_ids:
            # Get id.
            filter_dict['skip'] = skip
            filter_dict['limit'] = limit
            if log:
                print('get_generator: filter_dict_2={filter_dict}.'.format(
                    filter_dict=filter_dict))
            output = mlmt_client.ids.get_ids(filter_dict=filter_dict).result()
            if output is None:
                raise MongoQueryException(
                    'Failed to query ids. output not returned.'
                )
            status = output['status']
            if status != '200 OK':
                raise MongoQueryException(
                    ('Failed to query ids. Status: {status},'
                     ' Errors: {errors}').format(
                        status=status,
                        errors=output['errors']
                    )
                )
            ids = list(map(lambda d: d['_id'], output['ids']))
            if log:
                print('get_generator: ids={ids}.'.format(ids=ids))
            if ids == []:
                # Stop generator.
                break
            if len(ids) > limit:
                raise Exception(('Too many ids ({num}) returned.'
                                 ' Limit was {limit}.').format(
                    num=len(ids), limit=limit))
        while ids != []:
            # Now, get item by id.
            id_filter_dict = {
                '_id': ids[0],
                'collection_name': filter_dict['collection_name'],
                'return_type': filter_dict['return_type']
            }
            id_filter_output = mlmt_client.ids.get_by_id(
                filter_dict=id_filter_dict).result()
            if id_filter_output is None:
                raise MongoQueryException(
                    ('Failed to query items.'
                     ' id_filter_output not returned.')
                )
            status = id_filter_output['status']
            if status != '200 OK':
                if log:
                    print('get_generator:\n' + id_filter_output['trace'])
                raise MongoQueryException(
                    ('Failed to query items. Status: {status},'
                     ' Errors: {errors}').format(
                        status=status,
                        errors=id_filter_output['errors']
                    )
                )
            item = id_filter_output['item']
            yield item
            # Move up the list
            ids.pop(0)
        if already_have_ids:
            # We have already iterated through all the ids.
            # No need to do the loop again.
            break
        skip += limit
