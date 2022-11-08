"""
file_utils.py
Utilities for files related functions
"""

import tarfile
import os 
import glob

def is_within_directory(directory, target):
   """Check if the member target is within the directory"""

   abs_directory = os.path.abspath(directory)
   abs_target = os.path.abspath(target)

   prefix = os.path.commonprefix([abs_directory, abs_target])

   return prefix == abs_directory

def safe_extract(tar, path=".", members=None, numeric_owner=False):
   """Fix the vulnerability of the path traversal attact in extract() and 
   extrall() functions.
   
   @see bugs -  CVE-2007-4559
   https://www.trellix.com/en-us/about/newsroom/stories/research/tarfile-exploiting-the-world.html

   Args:
      tar: An input location of the tgz file as string
      path: Output location path
      members: members of the subset list returned by getmembers()
      numberic_owner: if True, use the uid and gid number from the tar file to set owner/group of the extracted files 
   """
   for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not is_within_directory(path, member_path):
             raise Exception("Attempted Path Traversal in Tar File")

   tar.extractall(path, members, numeric_owner=numeric_owner)
