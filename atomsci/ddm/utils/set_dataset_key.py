from atomsci.ddm.utils import checksum_utils as cu
from atomsci.ddm.utils import model_file_reader as mfr


def set_dataset_key(tar_path, new_dataset_key, result_tar_path,
    ignore_hash=False):

    reader = mfr.ModelFileReader(tar_path)

    # check if the new dataset_key is the exact same as
    # the training data
    if not ignore_hash:
        old_checksum = reader.get_dataset_hash()
        new_checksum = cu.create_checksum(new_dataset_key)

        assert old_checksum == new_checksum, \
            f'Check sum for {new_dataset_key}, does not match' \
            ' saved dataset_key.'

    reader.set_dataset_key(new_dataset_key)

    reader.write_new_tar(result_tar_path)