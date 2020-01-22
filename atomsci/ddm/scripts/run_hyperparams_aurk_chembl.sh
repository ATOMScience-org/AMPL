#!/bin/bash
#SBATCH -A baasic
#SBATCH -N 1
#SBATCH -p partition=surface
#SBATCH -t 24:00:00
#SBATCH -p pbatch
#SBATCH --export=ALL
#SBATCH -D /p/lustre1/kmelough/aurk_pilot_public/slurm_files
start=`date +%s`
ddm_dir="/g/g14/kmelough/git/AMPL/atomsci/ddm"

/usr/mic/bio/anaconda3/bin/python $ddm_dir/utils/hyperparam_search_wrapper.py --config_file $ddm_dir/config_files/sol_chembl_scaffold_ecfp_nn_hyper_config1.json
/usr/mic/bio/anaconda3/bin/python $ddm_dir/utils/hyperparam_search_wrapper.py --config_file $ddm_dir/config_files/sol_chembl_scaffold_mordred_nn_hyper_config1.json
/usr/mic/bio/anaconda3/bin/python $ddm_dir/utils/hyperparam_search_wrapper.py --config_file $ddm_dir/config_files/sol_chembl_scaffold_graphconv_nn_hyper_config1.json



#/usr/mic/bio/anaconda3/bin/python $ddm_dir/utils/hyperparam_search_wrapper.py --config_file $ddm_dir/config_files/bsep_chembl_scaffold_graphconv_nn_hyper_config1.json
#/usr/mic/bio/anaconda3/bin/python $ddm_dir/utils/hyperparam_search_wrapper.py --config_file $ddm_dir/config_files/herg_chembl_scaffold_graphconv_nn_hyper_config1.json
#/usr/mic/bio/anaconda3/bin/python $ddm_dir/utils/hyperparam_search_wrapper.py --config_file $ddm_dir/config_files/bsep_chembl_scaffold_mordred_nn_hyper_config1.json
#/usr/mic/bio/anaconda3/bin/python $ddm_dir/utils/hyperparam_search_wrapper.py --config_file $ddm_dir/config_files/herg_chembl_scaffold_mordred_nn_hyper_config1.json
#/usr/mic/bio/anaconda3/bin/python $ddm_dir/utils/hyperparam_search_wrapper.py --config_file $ddm_dir/config_files/bsep_chembl_scaffold_ecfp_nn_hyper_config1.json
#/usr/mic/bio/anaconda3/bin/python $ddm_dir/utils/hyperparam_search_wrapper.py --config_file $ddm_dir/config_files/herg_chembl_scaffold_ecfp_nn_hyper_config1.json
#/usr/mic/bio/anaconda3/bin/python $ddm_dir/utils/hyperparam_search_wrapper.py --config_file $ddm_dir/config_files/aurka_chembl_scaffold_graphconv_nn_hyper_config1.json
#/usr/mic/bio/anaconda3/bin/python $ddm_dir/utils/hyperparam_search_wrapper.py --config_file $ddm_dir/config_files/aurkb_chembl_scaffold_graphconv_nn_hyper_config1.json
#/usr/mic/bio/anaconda3/bin/python $ddm_dir/utils/hyperparam_search_wrapper.py --config_file $ddm_dir/config_files/aurka_chembl_scaffold_mordred_nn_hyper_config1.json
#/usr/mic/bio/anaconda3/bin/python $ddm_dir/utils/hyperparam_search_wrapper.py --config_file $ddm_dir/config_files/aurkb_chembl_scaffold_mordred_nn_hyper_config1.json
#/usr/mic/bio/anaconda3/bin/python $ddm_dir/utils/hyperparam_search_wrapper.py --config_file $ddm_dir/config_files/aurka_chembl_scaffold_ecfp_nn_hyper_config1.json
#/usr/mic/bio/anaconda3/bin/python $ddm_dir/utils/hyperparam_search_wrapper.py --config_file $ddm_dir/config_files/aurkb_chembl_scaffold_ecfp_nn_hyper_config1.json


end=`date +%s`
runtime=$((end-start))
echo "runtime: " $runtime
