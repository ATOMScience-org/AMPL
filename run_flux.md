# Flux

## Useful Links

* [Introduction to Flux](https://hpc-tutorials.llnl.gov/flux/)
* [Batch System Cross-Reference Guides](https://hpc.llnl.gov/banks-jobs/running-jobs/batch-system-cross-reference-guides)

## How To Run

1. If running AMPL, follow the [steps](https://github.com/ATOMScience-org/AMPL/tree/master#installation-quick-summary) to create an environment to run and build.

2. Activate your environment

### There are two ways to run Flux:

1. [Batch jobs](https://hpc-tutorials.llnl.gov/flux/section3/)
   
2. Interactive

   a. To start an interactive session:

   ```
   flux alloc -N1 -t 4h        # flux session, 1 node, 4 hours for example
   ```

   b. Use the session to run your jobs.

   If using *Nvidia*
   ```
   module load cuda/11.3.0     # load the Cuda
   ```

   If using *AMD*

   ```
   module load rocm/5.2.3      # load the ROCm module
   ```

   > ***Note***:
   > *To find out what package versions are available to your run machine*

   ```
   module avail [cuda|rocm]    # find available cuda or rocm versions
   ```

   c. Run your jobs/code