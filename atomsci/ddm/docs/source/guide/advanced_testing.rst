.. _advancd_testing:

.. _advanced_testing:

Advanced Testing
================

Running all tests
-----------------
To run the full set of tests, use Pytest from the test directory:
::

    source atomsci/bin/activate # activate your atomsci `pip env``
    cd $AMPL_HOME/atomsci/ddm/test # your ampl repo
    pytest
 

Running SLURM tests
-------------------
Several of the tests take some time to fit. These tests can be submitted to a SLURM cluster as a batch job. Example general SLURM submit scripts are included as pytest_slurm.sh.
::

    source atomsci/bin/activate
    cd $AMPL_HOME/atomsci/ddm/test/integrative/delaney_NN
    sbatch pytest_slurm.sh
    cd ../../../..
    cd $AMPL_HOME/atomsci/ddm/test/integrative/wenzel_NN
    sbatch pytest_slurm.sh

Running tests without internet access
-------------------------------------
**AMPL** works without internet access. Curation, fitting, and prediction do not require internet access.

However, the public datasets used in tests and examples are not included in the repo due to licensing concerns. These are automatically downloaded when the tests are run.

If a system does not have internet access, the datasets will need to be downloaded before running the tests and examples. From a system with internet access, run the following shell script to download the public datasets. Then, copy the AMPL directory to the offline system.
::

    cd atomsci/ddm/test
    bash download_datset.sh
    cd ../../..
    # Copy AMPL directory to offline system
