"""file_utils.py
Utilities for file related functions
"""

import os 

def is_within_directory(directory, target):
   """Check if the member target is within the directory"""

   abs_directory = os.path.abspath(directory)
   abs_target = os.path.abspath(target)

   prefix = os.path.commonprefix([abs_directory, abs_target])

   return prefix == abs_directory

def safe_extract(tar, path=".", members=None, numeric_owner=False):
   """Fix the vulnerability of the path traversal attack in extract() and
   extractall() functions.

   @see bugs -  CVE-2007-4559
   https://www.trellix.com/en-us/about/newsroom/stories/research/tarfile-exploiting-the-world.html

   Args:
      tar (tarfile.TarFile): A TarFile object representing an open tar archive.
      path: Output file path where the archive is to be extracted.
      members: Relative paths for the members of the tar archive to be extracted.
      numeric_owner: If True, use the uid and gid number from the tar file to set owner/group of the extracted files.
   """
   for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not is_within_directory(path, member_path):
             raise Exception("Attempted Path Traversal in Tar File")

   tar.extractall(path, members, numeric_owner=numeric_owner)
