# AMPL Pip Build README

AMPL provides different pip installation files for different environments and usages:

* clients_requirements.txt : `clients_requirements.txt` and `requirements.txt` are used for LLNL LC environment build.

* cpu_requirements.txt : minimal installation to run AMPL with CPU-only support.

* cuda_requirements.txt : minimal installation to run AMPL with CUDA-enabled GPUs.

* docker_requirements.txt : is used for AMPL docker image build.

* requirements.txt : complete build for both LLNL internal, external developers.

* readthedocs_requirements.txt : is used for readthedocs build.

* rocm_requirements.txt : minimal installation to run AMPL with AMD ROCm GPUs. (**work in progress**)

* simple_requirements.txt : basic installation to run AMPL.

Please refer the [README](https://github.com/ATOMScience-org/AMPL#create-pip-env) for more details on the different build options.
