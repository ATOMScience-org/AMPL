import tarfile
import os
import json
import argparse
import sys
import numpy as np

from atomsci.ddm.pipeline import parameter_parser as parse

def get_multiple_models_metadata(*args):
    """A function that takes model tar.gz file(s) and extract the metadata (and if applicable, model metrics)

    Args:
        *args: Variable length argument list of model tar.gz file(s)

    Returns:
        a list of models' most important model parameters and metrics. or an empty array if it fails to parse the input file(s).

    Exception:
        IOError: Problem access the file or if fails to parse the input file to an AMPL model

    """
    metadata_list = []
    for arg in args:
        try:
            metadata = ModelFileReader(arg).get_model_info()
            metadata_list.append(metadata)
        except:
            raise IOError("Problem access the file(s) or not AMPL model tarball(s).")
            
    return metadata_list
    
class ModelFileReader:
    """A class to encapsulate a model's metadata that you might want read out from a folder.
    Like read version number, get the dataset key, split uuid etc of a model.

    Attributes:
        Set in __init__:
            data_file_path (str): a model data file or a directory that contains the model

    """
    def __init__(self, data_file_path):
        """Constructor for the model object

        Args:
            data_file (str):  model data file
            model_tarpath (str): path to model tar file. The default is None.

        Exceptions:
            IOError: if the input file not in a valid json format
            TypeError: if the input file failed to parse into an AMPL model

        """
        self.model_path = data_file_path
        
        if os.path.isdir(data_file_path):
            self.metadata_path = os.path.join(self.model_path, 'model_metadata.json')
            if not os.path.exists(self.metadata_path):
                raise IOError(f"Could not find 'model_metadata.json' from {self.metadata_path}")
            
            with open(self.metadata_path, 'r') as data_file:
                self.metadata_dict = json.load(data_file)
                self.pparams = parse.wrapper(self.metadata_dict)
        else:
            with tarfile.open(data_file_path, 'r:gz') as tarball:
                try:
                    meta_info = tarball.getmember('./model_metadata.json')
                except KeyError:
                    print(f"{tarpath} is not an AMPL model tarball")
                    return {}
                with tarball.extractfile(meta_info) as meta_fd:
                    self.metadata_dict = json.loads(meta_fd.read())
                    self.pparams = parse.wrapper(self.metadata_dict)
                
    def get_descriptor_type(self):
        """Returns:
            (str): model descriptor type

        """
        descriptor_specific = self.metadata_dict.get('descriptor_specific')
        if descriptor_specific is None:
            return 'NA'
        descriptor_type = descriptor_specific.get('descriptor_type')
        return descriptor_type

    def get_model_parameters(self):
        """Returns:
            (str): model parameters

        """
        return self.metadata_dict.get("model_parameters")

    def get_model_uuid(self):
        """Returns:
            (str): model uuid

        """
        return self.metadata_dict.get("model_uuid")

    def get_version(self):
        """Returns:
            (str): model version

        """
        version = self.get_model_parameters().get("ampl_version", 'probably 1.0.0')
        return version

    def get_featurizer(self):
        """Returns:
            (str): model featurizer

        """
        featurizer = self.get_model_parameters().get('featurizer')
        return featurizer

    def get_model_type(self):
        """Returns:
            (str): model type

        """
        return self.get_model_parameters().get('model_type')

    def get_training_dataset(self):
        """Returns:
            (str): model training dataset

        """
        return self.metadata_dict.get('training_dataset')

    def get_dataset_key(self):
        """Returns:
            (str): model dataset key

        """
        return self.get_training_dataset().get('dataset_key')

    def get_split_csv(self):
        """Returns:
            (str): model split csv

        """
        no_csv = os.path.splitext(self.get_dataset_key())[0]
        return f'{no_csv}_{self.get_split_strategy()}_{self.get_splitter()}_{self.get_split_uuid()}.csv'

    def get_splitting_parameters(self):
        """Returns:
            (str): model splitting parameters

        """
        return self.metadata_dict.get('splitting_parameters')

    def get_split_uuid(self):
        """Returns:
            (str): model split_uuid

        """
        split_uuid = self.get_splitting_parameters().get('split_uuid')
        return split_uuid

    def get_split_strategy(self):
        """Returns:
            (str): model split strategy

        """
        split_strat = self.get_splitting_parameters().get('split_strategy')
        return split_strat

    def get_splitter(self):
        """Returns:
            (str): model splitter

        """
        splitter = self.get_splitting_parameters().get('splitter')
        return splitter

    def get_id_col(self):
        """Returns:
            (str): model id column

        """
        return self.get_training_dataset().get('id_col')

    def get_smiles_col(self):
        """Returns:
            (str): model smile columns

        """
        return self.get_training_dataset().get('smiles_col')

    def get_response_cols(self):
        """Returns:
            (str): model response columns

        """
        return self.get_training_dataset().get('response_cols')
    
    def get_model_info(self):
        """Extract the model metadata (and if applicable, model metrics)

        Returns:
            a dictionary of the most important model parameters and metrics.
        """
        model_dict = dict(
            model_path = self.model_path,
            model_uuid = self.pparams.model_uuid,
            model_type = self.pparams.model_type,
            pred_type = self.pparams.prediction_type,
            response_cols ='; '.join(self.pparams.response_cols),
            dataset_key = self. pparams.dataset_key,
            splitter = self.pparams.splitter,
            featurizer = self.pparams.featurizer,
            )
        
        if self.pparams.featurizer in ['computed_descriptors', 'descriptors']:
            model_dict['features'] = self.pparams.descriptor_type
        else:
            model_dict['features'] = self.pparams.featurizer
            
        if self.pparams.datastore:
            model_dict['bucket'] = self.pparams.bucket
            ds_client = dsf.config_client()
            try:
                ds_dset = ds_client.ds_datasets.get_bucket_dataset(bucket_name=bucket, dataset_key=dskey).result()
                model_dict['dataset_available'] = True
            except bravado.exception.HTTPNotFound:
                model_dict['dataset_available'] = False
        else:
            model_dict['bucket'] = np.nan
            model_dict['dataset_available'] = os.path.exists(self.pparams.dataset_key)
        
        if self.pparams.prediction_type == 'regression':
            metric_type = 'r2_score'
        else:
            metric_type = 'roc_auc_score'
            
        model_dict['metric_type'] = metric_type
        
        try:
            metrics = self.metadata_dict['training_metrics']
            for metric in metrics:
                if metric['label'] != 'best':
                    continue
                metric_name = f"{metric['subset']}_metric"
                model_dict[metric_name] = metric['prediction_results'][metric_type]
        except KeyError:
            model_dict['train_metric'] = np.nan
            model_dict['valid_metric'] = np.nan
            model_dict['test_metric'] = np.nan
        
        return model_dict

#----------------
# main
#----------------
def main(argv):

    # input file/dir (required)
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='input model directory/file')

    args = parser.parse_args()

    model = ModelFileReader(args.input)

    if model is not None:
        print('data set key: ', model.get_dataset_key())
        print('split uuid: ', model.get_split_uuid())
        print('model uuid: ', model.get_model_uuid())
        
if __name__ == "__main__":
   main(sys.argv[1:])
