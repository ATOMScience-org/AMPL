#
# setup.py
#

from setuptools import setup, find_namespace_packages
import os
import glob

PACKAGE = 'atomsci'

here = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(here, os.pardir))
script_files = glob.glob("scripts/*")

setup(
    name='{}-ampl'.format(PACKAGE),
    # install_requires=['distribute'],
    include_package_data=True,
    version=open('VERSION').read().strip(),
    description='{} AMPL Python Package'.format(PACKAGE),
    zip_safe=False,
    data_files=[],
    packages=find_namespace_packages(include=['atomsci.*']),
    scripts=script_files,
    install_requires=[
    ],
    entry_points={
    },
)