import atomsci.ddm.utils.patch_model_dataset_key as pmdk
import atomsci.ddm.utils.model_file_reader as mfr
import os
import inspect
import pytest


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

def test_patch_model_dataset_key():
    model_path = os.path.join(
        currentdir,
        '../../examples/tutorials/dataset/SLC6A3_models/SLC6A3_Ki_curated_model_b24a2887-8eca-43e2-8fc2-3642189d2c94.tar.gz'
    )
    new_path = os.path.join(
        currentdir,
        'SLC6A3_Ki_curated_model_b24a2887-8eca-43e2-8fc2-3642189d2c94.tar.gz'
    )
    dataset_path = os.path.realpath(os.path.join(
        currentdir,
        '../../examples/tutorials/dataset/SLC6A3_IC50_curated.csv'
    ))

    return_value = pmdk.patch_model_dataset_key(
        model_path=model_path,
        new_model_path=new_path,
        dataset_path=dataset_path,
        require_hash_match=False
    )

    new_dataset_key = mfr.ModelFileReader(new_path).get_dataset_key()

    assert new_dataset_key == dataset_path
    assert return_value == 0
    os.remove(new_path)

def test_hash_mismatch():
    model_path = os.path.join(
        currentdir,
        '../../examples/tutorials/dataset/SLC6A3_models/SLC6A3_Ki_curated_model_b24a2887-8eca-43e2-8fc2-3642189d2c94.tar.gz'
    )
    new_path = os.path.join(
        currentdir,
        'SLC6A3_Ki_curated_model_b24a2887-8eca-43e2-8fc2-3642189d2c94.tar.gz'
    )
    dataset_path = os.path.realpath(os.path.join(
        currentdir,
        '../../examples/tutorials/dataset/SLC6A3_IC50_curated.csv'
    ))

    return_value = pmdk.patch_model_dataset_key(
        model_path=model_path,
        new_model_path=new_path,
        dataset_path=dataset_path,
        require_hash_match=True
    )

    assert return_value == 1

def test_bad_tar_file():
    model_path = os.path.join(
        currentdir,
        '../test_datasets/bad_model_tar.tar.gz'
    )
    new_path = os.path.join(
        currentdir,
        'SLC6A3_Ki_curated_model_b24a2887-8eca-43e2-8fc2-3642189d2c94.tar.gz'
    )
    dataset_path = os.path.join(
        currentdir,
        '../../examples/tutorials/dataset/SLC6A3_IC50_curated.csv'
    )

    with pytest.raises(ValueError) as e:
        _ = pmdk.patch_model_dataset_key(
            model_path=model_path,
            new_model_path=new_path,
            dataset_path=dataset_path,
            require_hash_match=False
        )
    assert e.type == ValueError

def test_check_data_accessibility_tar():
    # just a tar file
    model_path = os.path.join(
        currentdir,
        '../../examples/tutorials/dataset/SLC6A3_models/SLC6A3_Ki_curated_model_b24a2887-8eca-43e2-8fc2-3642189d2c94.tar.gz'
    )

    dataset_info = pmdk.check_data_accessibility(model_path=model_path)
    assert len(dataset_info) == 1

def test_check_data_accessibility_folder():
    # directory with tarballs
    model_path = os.path.join(
        currentdir,
        '../../examples/tutorials/dataset/SLC6A3_models/'
    )

    dataset_info = pmdk.check_data_accessibility(model_path=model_path)
    assert len(dataset_info) == 4

def test_check_data_accessibility_bad():
    # directory with bad tarball
    model_path = currentdir

    dataset_info = pmdk.check_data_accessibility(model_path=model_path, verbose=True)
    assert len(dataset_info) == 0

if __name__ == '__main__':
    test_check_data_accessibility_tar()
    test_check_data_accessibility_bad()
    test_check_data_accessibility_folder()