### To Run on Flux on AMD

1. Follow the [steps](https://github.com/ATOMScience-org/AMPL/tree/master#installation-quick-summary) to create an environment to run and build AMPL.

2. Activate your environment

3. Start a FLUX interactive session

```
flux alloc -N1 -t 4h        # flux session, 1 node, 4 hours for example
module load rocm/5.2.3      # load the ROCm module on AMD 
....                        # run your jobs
```
