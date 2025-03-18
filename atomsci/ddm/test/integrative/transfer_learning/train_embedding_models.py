import os
import atomsci.ddm.pipeline.model_pipeline as mp
import atomsci.ddm.pipeline.parameter_parser as parse
import glob
import json

def train_models():
    jsons = glob.glob(os.path.join(os.path.dirname(__file__), '*.json'))
    for json_file in jsons:
        with open(json_file, 'r') as js:
            params = json.load(js)

        result_dir = os.path.join(os.path.dirname(__file__),
            'test_embedding')
        dskey = os.path.join(os.path.dirname(__file__),
            '../../test_datasets/delaney-processed_curated_fit.csv')
        params['result_dir'] = result_dir
        params['dataset_key'] = dskey

        pparams = parse.wrapper(params)

        model_pipeline = mp.ModelPipeline(pparams)
        model_pipeline.train_model()
        print(json_file)
        print(model_pipeline.params.model_uuid)

    # After this manually move files from result_dir to
    # expected location and rename them.

if __name__ == '__main__':
    train_models()