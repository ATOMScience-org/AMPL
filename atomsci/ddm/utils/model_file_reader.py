import tempfile
import tarfile
import os
import json
import argparse
import sys

class ModelFolder:
    '''
    This encapsulates somethings you might want read out of a folder.

    Like read version number get the dataset key, split uuid etc of a model
    '''
    def __init__(self, dir_path):
        self.dir_path = dir_path

        self.metadata_path = os.path.join(self.dir_path, 'model_metadata.json')
        assert os.path.exists(self.metadata_path), f"Could not find {self.metadata_path}"

        with open(self.metadata_path, 'r') as data_file:
            self.metadata_dict = json.load(data_file)

    def get_descriptor_type(self):
        descriptor_specific = self.metadata_dict.get('descriptor_specific')
        if descriptor_specific is None:
            return 'NA'
        descriptor_type = descriptor_specific.get('descriptor_type')
        return descriptor_type

    def get_model_parameters(self):
        return self.metadata_dict.get("model_parameters")

    def get_model_uuid(self):
        return self.metadata_dict.get("model_uuid")

    def get_version(self):
        version = self.get_model_parameters().get("ampl_version", 'probably 1.0.0')
        return version

    def get_featurizer(self):
        featurizer = self.get_model_parameters().get('featurizer')
        return featurizer

    def get_model_type(self):
        return self.get_model_parameters().get('model_type')

    def get_training_dataset(self):
        return self.metadata_dict.get('training_dataset')

    def get_dataset_key(self):
        return self.get_training_dataset().get('dataset_key')

    def get_split_csv(self):
        no_csv = os.path.splitext(self.get_dataset_key())[0]
        return f'{no_csv}_{self.get_split_strategy()}_{self.get_splitter()}_{self.get_split_uuid()}.csv'

    def get_splitting_parameters(self):
        return self.metadata_dict.get('splitting_parameters')

    def get_split_uuid(self):
        split_uuid = self.get_splitting_parameters().get('split_uuid')
        return split_uuid

    def get_split_strategy(self):
        split_strat = self.get_splitting_parameters().get('split_strategy')
        return split_strat

    def get_splitter(self):
        splitter = self.get_splitting_parameters().get('splitter')
        return splitter

    def get_id_col(self):
        return self.get_training_dataset().get('id_col')

    def get_smiles_col(self):
        return self.get_training_dataset().get('smiles_col')

    def get_response_cols(self):
        return self.get_training_dataset().get('response_cols')

class ModelTar(ModelFolder):
    '''
    Same thing as a ModelFolder, except it starts off as a tarball
    '''
    def __init__(self, tar_path):
        self.tar_path = tar_path
        tmpdir = tempfile.mkdtemp()
            
        model_fp = tarfile.open(self.tar_path, mode='r:gz')
        model_fp.extractall(path=tmpdir)
        model_fp.close()

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

    tar_model = ModelTar(finput)
    print('data set key ', tar_model.get_dataset_key())
    print('split uuid ', tar_model.get_split_uuid())
    print('model uuid ', tar_model.get_model_uuid())
    print('version ', tar_model.get_version())
    print('model ', tar_model.get_model_type())

if __name__ == "__main__":
   main(sys.argv[1:])
