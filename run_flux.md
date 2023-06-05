# Flux

`Flux` is a new resource management and scheduling system developed at `LLNL`. It has replaced `SLURM` as the native scheduler on `LLNL` clusters such as `corona` and `tioga`, and will eventually do so on other clusters. You need to learn basic `Flux` commands in order to run batch jobs and make use of `GPU` nodes on `corona` and `tioga`.

## Useful Links

* [Introduction to Flux](https://hpc-tutorials.llnl.gov/flux/)
* [Flux Documentation](https://flux-framework.readthedocs.io/en/latest/)
* [Flux FAQ](https://flux-framework.readthedocs.io/en/latest/faqs.html)
* [Batch System Cross-Reference Guides](https://hpc.llnl.gov/banks-jobs/running-jobs/batch-system-cross-reference-guides)

## How To Run

1. If running AMPL, follow the [steps](./README.md#Install) to create an environment to run and build.

2. Activate your environment

### There are two ways to run Flux:

1. [Batch jobs](https://hpc-tutorials.llnl.gov/flux/section3/)
   
2. Interactive

   a. To start an interactive session:

   ```
   flux alloc -N1 -t 4h        # flux session, 1 node, 4 hours for example
   ```

   b. Use the session to run your jobs.

   * If using *Nvidia*
   ```
   module load cuda/11.3.0     # load the Cuda
   ```

   <a name="amd"></a>
   * If using *AMD*

   ```
   module load rocm/5.2.3      # load the ROCm module
   ```

   > ***Note***:
   > *To find out what package versions are available to your run machine*

   ```
   module avail [cuda|rocm]    # find available cuda or rocm versions
   ```

   c. Run your jobs/code
