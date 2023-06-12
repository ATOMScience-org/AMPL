import tempfile
import tarfile
import os
import json
import argparse
import sys
import atomsci.ddm.utils.file_utils as futils

class ModelFolder:
    """Base class to encapsulate a model's metadata that you might want read out from a folder.
    Like read version number, get the dataset key, split uuid etc of a model.

    Attributes:
        Set in __init__:
            dir_path (str): directory path that contains model data

    """
    def __init__(self, dir_path):
        """Constructor for the model object

        Args:
            dir_path (str):  directory path that contains model data

        Raises:
            Exception: if `model_metadata.json` not found in the directory

        """
        self.dir_path = dir_path

        self.metadata_path = os.path.join(self.dir_path, 'model_metadata.json')
        if not os.path.exists(self.metadata_path):
            raise IOError(f"Could not find 'model_metadata.json' from {self.metadata_path}")

        with open(self.metadata_path, 'r') as data_file:
            self.metadata_dict = json.load(data_file)

    def get_descriptor_type(self):
        """
        Returns:
            (str): model descriptor type

        """
        descriptor_specific = self.metadata_dict.get('descriptor_specific')
        if descriptor_specific is None:
            return 'NA'
        descriptor_type = descriptor_specific.get('descriptor_type')
        return descriptor_type

    def get_model_parameters(self):
        """
        Returns:
            (str): model parameters

        """
        return self.metadata_dict.get("model_parameters")

    def get_model_uuid(self):
        """
        Returns:
            (str): model uuid

        """
        return self.metadata_dict.get("model_uuid")

    def get_version(self):
        """
        Returns:
            (str): model version

        """
        version = self.get_model_parameters().get("ampl_version", 'probably 1.0.0')
        return version

    def get_featurizer(self):
        """
        Returns:
            (str): model featurizer

        """
        featurizer = self.get_model_parameters().get('featurizer')
        return featurizer

    def get_model_type(self):
        """
        Returns:
            (str): model type

        """
        return self.get_model_parameters().get('model_type')

    def get_training_dataset(self):
        """
        Returns:
            (str): model training dataset

        """
        return self.metadata_dict.get('training_dataset')

    def get_dataset_key(self):
        """
        Returns:
            (str): model dataset key

        """
        return self.get_training_dataset().get('dataset_key')

    def get_split_csv(self):
        """
        Returns:
            (str): model split csv

        """
        no_csv = os.path.splitext(self.get_dataset_key())[0]
        return f'{no_csv}_{self.get_split_strategy()}_{self.get_splitter()}_{self.get_split_uuid()}.csv'

    def get_splitting_parameters(self):
        """
        Returns:
            (str): model splitting parameters

        """
        return self.metadata_dict.get('splitting_parameters')

    def get_split_uuid(self):
        """
        Returns:
            (str): model split_uuid

        """
        split_uuid = self.get_splitting_parameters().get('split_uuid')
        return split_uuid

    def get_split_strategy(self):
        """
        Returns:
            (str): model split strategy

        """
        split_strat = self.get_splitting_parameters().get('split_strategy')
        return split_strat

    def get_splitter(self):
        """
        Returns:
            (str): model splitter

        """
        splitter = self.get_splitting_parameters().get('splitter')
        return splitter

    def get_id_col(self):
        """
        Returns:
            (str): model id column

        """
        return self.get_training_dataset().get('id_col')

    def get_smiles_col(self):
        """
        Returns:
            (str): model smile columns

        """
        return self.get_training_dataset().get('smiles_col')

    def get_response_cols(self):
        """
        Returns:
            (str): model response columns

        """
        return self.get_training_dataset().get('response_cols')

class ModelTar(ModelFolder):
    '''
    Same thing as a ModelFolder, except it starts off as a tarball

        Args:
            tar_path (str):  directory path that contains a model tar file

        Raises:
            Exception: if the input does not ends with tar.gz

    '''
    def __init__(self, tar_path):
        if not tar_path.endswith('tar.gz'):
            raise IOError(f"Expect tar.gz file from {tar_path}")

        self.tar_path = tar_path
        tmpdir = tempfile.mkdtemp()
            
        with tarfile.open(tar_path, mode='r:gz') as tar:
            futils.safe_extract(tar, path=tmpdir)

        super().__init__(tmpdir)


#----------------
# main
#----------------
def main(argv):

    # input file/dir (required)
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='input model directory/file')

    args = parser.parse_args()

    finput = args.input

    model_folder = None

    if os.path.isdir(finput):
        model_folder = ModelFolder(finput)
    elif os.path.isfile(finput):
        model_folder = ModelTar(finput)

    if model_folder is not None:
        print('data set key ', model_folder.get_dataset_key())
        print('split uuid ', model_folder.get_split_uuid())
        print('model uuid ', model_folder.get_model_uuid())
        print('version ', model_folder.get_version())
        print('model ', model_folder.get_model_type())
        print('split csv ', model_folder.get_split_csv())
        print('split params ', model_folder.get_splitting_parameters())
        print('smiles col ', model_folder.get_smiles_col())

if __name__ == "__main__":
   main(sys.argv[1:])
