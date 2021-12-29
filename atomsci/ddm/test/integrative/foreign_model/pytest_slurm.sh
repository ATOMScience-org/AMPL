#!/bin/bash
#SBATCH -t 6:00:00
#SBATCH --export=ALL

echo "Start: " `date`
start=`date +%s`

test_directory=../../../examples/BSEP
echo "Test directory: " $test_directory
cd $test_directory

echo "\nWithout ADI"

./predict_bsep_inhibition.py -i data/small_test_data.csv -o small_test_output.csv --id_col compound_name --smiles_col base_rdkit_smiles --activity_col active

echo "\nWith ADI"

./predict_bsep_inhibition.py -i data/small_test_data.csv -o small_test_output.csv --id_col compound_name --smiles_col base_rdkit_smiles --activity_col active --ad_method z_score --ext_train_data data/morgan_warner_combined_bsep_data.csv

rm small_test_output.csv

echo "End: " `date`
end=`date +%s`

runtime=$((end-start))
echo "Wall time: " $runtime
