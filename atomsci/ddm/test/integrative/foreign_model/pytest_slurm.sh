#!/bin/bash
#SBATCH -t 6:00:00
#SBATCH --export=ALL

echo "Start: " `date`
start=`date +%s`

test_directory=`pwd`
echo "Test directory: " $test_directory
cd $test_directory

pytest

echo "End: " `date`
end=`date +%s`

runtime=$((end-start))
echo "Wall time: " $runtime