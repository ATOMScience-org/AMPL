## Flux

### Useful Links

* [Introduction to Flux](https://hpc-tutorials.llnl.gov/flux/)
   * [Batch jobs](https://hpc-tutorials.llnl.gov/flux/section3/)
   * Interactive (see #3 below)

* [Batch System Cross-Reference Guides](https://hpc.llnl.gov/banks-jobs/running-jobs/batch-system-cross-reference-guides)

### How To Run 

1. If running AMPL, follow the [steps](https://github.com/ATOMScience-org/AMPL/tree/master#installation-quick-summary) to create an environment to run and build.

2. Activate your environment

3. Start a Flux interactive session

```
flux alloc -N1 -t 4h        # flux session, 1 node, 4 hours for example
```

4.  Use the session to run your jobs. If on *Nvidia*
```
module load cuda/11.3.0     # load the Cuda
....                        # run your jobs
```

If on *AMD*

```
module load rocm/5.2.3      # load the ROCm module 
....                        # run your jobs
```

> ***Note***:
> *To find out what package versions are available to your run machine*

```
module avail [cuda|rocm]    # find available cuda or rocm versions
```
