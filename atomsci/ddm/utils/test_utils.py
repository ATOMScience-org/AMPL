import os

def relative_to_file(file_path, relative_path):
    """Useful for building aboslute paths relative to scripts file path.
    Used in integrative testing when relative paths to data files are needed.

    Arguments:
        file_path: This is __file__
        relative:path: This is something like ./example_file.csv

    Returns:
        This returns a realpath to example_file.csv
    """

    dir_name = os.path.dirname(file_path)
    return os.path.realpath(os.path.join(dir_name, relative_path))