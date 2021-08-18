#!/bin/bash

###
### Example script to run tests on curation code
###
### All the details are specified in the configuaration file (config_parser.ini)
###

python target_data_curation.py -config_file config_parser.ini


##
## the target names are different for chembl
## cat gene_lst_v1.txt | perl chmbl_map_help.pl > gene_lst_v1_chembl.txt
## then manually fix the column header
