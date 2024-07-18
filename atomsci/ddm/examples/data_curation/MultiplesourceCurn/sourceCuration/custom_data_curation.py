import matplotlib
matplotlib.use('Agg')
import pandas as pd
import os

import json

#import custom_config as cc
import atomsci.ddm.utils.pubchem_utils as pu
from os import path
from target_data_curation import AMPLDataset
from atomsci.ddm.utils import data_curation_functions as dcf
import numpy as np


# Using data_curation_functions
# Initial dataset downloaded org/record/173258#.XcXWHZJKhhE
# cd /usr/workspace/atom/excapedb/
# Look for Last-Modified and Length to check whether to download update DATE=date -Idate
#---
# wget -S https://zenodo.org/record/173258/files/pubchem.chembl.dataset4publication_inchi_smiles.tsv.xz?download=1 -O excape_data.csv.$DATE Last-Modified: 12/29/2016
# head -1 pubchem.chembl.dataset4publication_inchi_smiles.tsv > cyp3a4_excape.csv
# grep 'CYP3A4' pubchem.chembl.dataset4publication_inchi_smiles.tsv >> cyp3a4_excape.csv
#---
# grep CYP3A4 pubchem.chembl.dataset4publication_inchi_smiles.tsv > raw_data.txt
# head -1 pubchem.chembl.dataset4publication_inchi_smiles.tsv > header
# cat header raw_data.txt > cyp3a4_excape.csv
#---

class ExcapeActivityDump(AMPLDataset):
    """Class responsible for parsing and extracting data from the Excape-DB
       tsv data dump file

    """

    def set_columns(self,sec) :
        """Sets expected column names for input file and reads the table."""
        self.smiles_col = 'SMILES'
        self.base_smiles_col = 'base_rdkit_smiles'
        self.id_col = 'compound_id'
        self.standard_col = 'Standard Type'
        self.target_name_col = 'Target Name'
        self.target_id_col = 'Gene_Symbol'
        self.relation_col = 'relation'
        self.value_col = 'activity'
        self.date_col = 'Document Year'
        self.data_source_name=sec

    def __init__(self, parser=None, sec = None, raw_target_lst = None, dataset = None, df = None):
        super().__init__()
        """
        Initialize object


        Params:
            parser : pointer to the config file parser to access user specificied configuration settings
            sec : specifies where in the config file to retrieve settings from 
            raw_target_lst : list of gene targets to extract data for
            dataset : gives an identifier 
            df : dataframe storing the raw data to be extracted and curated

        """
        if dataset is not None and df is not None :
            self.set_columns(dataset.data_source_name)
            self.df = df
        else :
            self.set_columns(sec)
            filename = parser.check_get(sec,'activity_csv')
            self.df = pd.read_csv(filename,sep="\t",engine="python",error_bad_lines=False)
            #the reason to fill empty values is to track what is likely inactive measurements
            #self.df['pXC50'] = self.df['pXC50'].fillna(0)
            #we just ignore them for now
            self.df = self.df.dropna(subset=['pXC50'])
             
            # (Yaru) Remove inf in curated_df
            self.df = self.df[~self.df.isin([np.inf]).any(1)]
      
            # standardize column names
            self.df.rename( columns={ "pXC50" : self.value_col, "Ambit_InchiKey" : self.id_col }, inplace=True)
      
            self.df[self.relation_col] = ""
            self.df.loc[self.df.Activity_Flag == 'N',self.relation_col] = '<'
            self.df.loc[self.df.Activity_Flag == 'A',self.relation_col] = ''
      
            # FILTER ON 9606 (human)
            self.df=self.df[self.df.Tax_ID == 9606]
            self.df = self.df[self.df[self.target_id_col].isin(raw_target_lst)]



class DrugTargetCommonsActivityDump(AMPLDataset):
    """Class responsible for parsing and extracting data from the Drug Target Commons
       data dump file

    """
    def set_columns(self, sec ) :
        """Sets expected column names for input file and reads the table."""
        self.smiles_col = 'smiles'
        self.base_smiles_col = 'base_rdkit_smiles'
        self.id_col = 'compound_id'
        self.standard_col = 'standard_type'
        self.target_name_col = 'gene_names'
        self.target_id_col = 'gene_names'
        self.relation_col = 'relation'
        self.value_col = 'standard_value'
        self.date_col = ''
        self.tmp_dir = './' # where to cache retrieved SMILES strings
        self.data_source_name=sec

    def __init__(self, parser=None, sec = None, raw_target_lst = None, dataset = None, df = None):
        super().__init__()
        """
        Initialize object


        Params:
            parser : pointer to the config file parser to access user specificied configuration settings
            sec : specifies where in the config file to retrieve settings from 
            raw_target_lst : list of gene targets to extract data for
            dataset : gives an identifier 
            df : dataframe storing the raw data to be extracted and curated

        """
        if dataset is not None and df is not None :
            self.set_columns(dataset.data_source_name)
            self.df = df
        else :
            self.set_columns(sec)
            self.tmp_dir = parser.check_get(sec,'output_data_dir')
            filename = parser.check_get(sec,'activity_csv')
            self.smiles_file=parser.check_get(sec,'smiles_csv')
            self.df = pd.read_csv(filename,sep=",",engine="python",error_bad_lines=False)
            end_point_lst=parser.check_get(sec,"end_points").split(',')
            self.df.standard_type = self.df.standard_type.str.lower()
            end_point_lst =  [ x.lower() for x in end_point_lst ]
            #
            # might want to filter on human
            # data_curatoin_functions.filter_dtc_data
            #
            #TODO : SHOULD FILTER ON 9606 (human) -Jonathan, why isn't this done?
            self.df = self.df[self.df.gene_names.isin(raw_target_lst) &
                           ~(self.df.standard_inchi_key.isna()) &
                           self.df.standard_type.isin( end_point_lst ) &
                               (self.df.standard_units == 'NM') &
                           ~self.df.standard_value.isna() &
                           ~self.df.compound_id.isna() &
                           (self.df.wildtype_or_mutant != 'mutated') ]

      
            ####WARNING: I had to convert this explicitly to a floating point value!!!
            self.df[self.value_col]=self.df[self.value_col].astype(float)
      
            ## convert values to -log10 molar values (function assumes input is in NM units)
            self.df[self.value_col]=self.df[self.value_col].apply(dcf.ic50topic50)
      
            # (Yaru) Remove inf in curated_df
            self.df = self.df[~self.df.isin([np.inf]).any(1)]
      
            # we shouldn't have to do this, but compound_id is hard coded in the aggregate_data function
            # we need to use the inchikey as the compound id, since the normal compound id isn't always defined/available      
            # so rename column to something different and name inchikey column as the compound_id column
            self.df.rename( columns={"standard_relation" : self.relation_col }, inplace=True)
            self.df.rename( columns={self.id_col : "orig_compound_id" }, inplace=True)
            self.df.rename( columns={"standard_inchi_key" : self.id_col }, inplace=True)
      
          
    def add_base_smiles_col(self):
      """requires a specialized SMILES curation step as SMILES strings are stored separately"""
      targLst=self.df[self.target_name_col].unique().tolist()
      targLst.sort()
      targ_name='_'.join(targLst)
      if len(targ_name) >= 25 :
         targ_name='target'
         fileNameTooLong=True
      else :
         fileNameTooLong=False
      myList=self.df[self.id_col].unique().tolist()
      #
      #TODO: Need to make this caching scheme user defined
      # now just hardcoded to write to current directory
      # Retrieve SMILES strings for compounds through PUBCHEM web interface.
      # THIS is slow so it should only be done once and then cached to file
      tmp_dir=self.tmp_dir
      ofile=tmp_dir+targ_name+'_dtc_smiles_raw.csv'
      if not path.exists(ofile):
         if not path.exists(self.smiles_file) :
            print("download from PubChem ")
            ## smiles are stored in 'smiles' column in returned dataframe
            save_smiles_df,fail_lst,discard_lst=pu.download_smiles(myList)
            save_smiles_df.to_csv(ofile, index=False)
         else :
            print("retrieve SMILES from predownloaded file",self.smiles_file)
            # save_smiles_df=pd.read_csv(self.smiles_file)
            # save_smiles_df.rename( columns={"inchikey" : self.id_col }, inplace=True)
            # save_smiles_df.to_csv(ofile, index=False)
            sed_cmd = f"sed 's/inchikey/{self.id_col}/' {self.smiles_file} > {ofile}"
            os.system(sed_cmd)
            save_smiles_df=pd.read_csv(ofile)
      else :
         print("Already download file",ofile)
         save_smiles_df=pd.read_csv(ofile)

      print("debug make sure smiles not empty",save_smiles_df.shape)
      #the file puts the SMILES string in quotes, which need to be removed
      save_smiles_df[self.smiles_col]=save_smiles_df[self.smiles_col].str.replace('"','')

      #need to join SMILES strings with main data_frame
      self.df=self.df.merge(save_smiles_df,on=self.id_col,suffixes=('_'+targ_name,'_'))
  
      # calculate base rdkit smiles and add them to the dataframe
      super().add_base_smiles_col()


class ChEMBLActivityDump(AMPLDataset):
    """Class responsible for parsing and extracting data from a ChEMBL json
       data dump file

    """
    def set_columns(self,sec) :
        """Sets expected column names for input file and reads the table."""
        self.smiles_col = 'smiles'
        self.base_smiles_col = 'base_rdkit_smiles'
        self.id_col = 'compound_id'
        self.standard_col = 'standard_type'
        self.target_id_col = 'gene_names'
        self.relation_col = 'relation'
        self.value_col = 'pAct'
        self.units = 'units'
        self.assay_id = 'assay_id'
        self.data_source_name=sec

    def __init__(self, parser=None, sec = None, raw_target_lst = None, dataset = None, df = None):
        super().__init__()
        """
        Initialize object


        Params:
            parser : pointer to the config file parser to access user specificied configuration settings
            sec : specifies where in the config file to retrieve settings from 
            raw_target_lst : list of gene targets to extract data for
            dataset : gives an identifier 
            df : dataframe storing the raw data to be extracted and curated

        """
        if dataset is not None and df is not None :
           self.set_columns(dataset.data_source_name)
           self.df = df
        else :
           self.set_columns(sec)
           mapgn=parser.check_get(sec,'target_mapping')
           self.target_dict = json.load(open(mapgn))
           end_point_lst=parser.check_get(sec,"end_points").split(',')
           end_point_lst =  [ x.lower() for x in end_point_lst ]

           filename=parser.check_get(sec,'activity_csv')
           dc=json.load(open(filename))
           target_lst=[]

           for val in raw_target_lst :
              target_lst.append( self.target_dict[val] )

           df_lst=[]
           for kv in raw_target_lst :
              tmp_df_lst=[]
              for cid in dc[kv].keys() :
                 lst=dc[kv][cid]['pAct']
                 for it in range(len(lst)) :
                     row={ self.id_col : cid, self.value_col : dc[kv][cid]['pAct'][it], self.relation_col : dc[kv][cid]['relation'][it],
                           self.smiles_col : dc[kv][cid]['smiles'], self.standard_col : dc[kv][cid]['type'][it], self.units : dc[kv][cid]['units'][it], self.assay_id : dc[kv][cid]['assay_id'][it]   }
                     tmp_df_lst.append(row)
              df=pd.DataFrame(tmp_df_lst)
              df[self.target_id_col] = self.target_dict[kv]
              df = df.dropna(subset=[self.value_col,self.standard_col,self.id_col])
              df_lst.append(df)

           self.df = pd.concat(df_lst)

           ## do we need to do any other filter/checks here, like units?
           self.df=self.df[(self.df.units.str.lower() == 'nm')]
           self.df.standard_type = self.df.standard_type.str.lower()
           self.df=self.df[self.df.standard_type.isin( end_point_lst ) ]

    def filter_task(self, target_id):
        """when the gene target label isn't standardized, need to return the gene target mapping"""
        return self.df[(self.df[self.target_id_col] == self.target_dict[target_id])],self.target_dict[target_id]

def convert_dtype(x):
    if not x:
        return 0.0
    try:
        return float(x)   
    except:        
        return 0.0

class GPCRChEMBLActivityDump(AMPLDataset):
    """Class responsible for parsing and extracting data from a custom ChEMBL formatted csv
       data dump file

    """
    def set_columns(self,sec) :
        """Sets expected column names for input file and reads the table."""
        ## complaining about mixed datatypes and I can't seem to fix it!
        self.smiles_col = 'Smiles'
        self.base_smiles_col = 'base_rdkit_smiles'
        self.id_col = 'compound_id'
        self.standard_col = 'relation'
        self.target_name_col = 'Target Name'
        self.target_id_col = 'gene_names'
        self.relation_col = 'relation'
        self.value_col = 'pChEMBL Value'
        self.date_col = 'Document Year'
        self.data_source_name=sec

    def __init__(self, parser=None, sec = None, raw_target_lst = None, dataset = None, df = None):
        super().__init__()
        """
        Initialize object

        Params:
            parser : pointer to the config file parser to access user specificied configuration settings
            sec : specifies where in the config file to retrieve settings from 
            raw_target_lst : list of gene targets to extract data for
            dataset : gives an identifier 
            df : dataframe storing the raw data to be extracted and curated

        """
        if dataset is not None and df is not None :
           self.set_columns(dataset.data_source_name)
           self.df = df
        else :
           self.set_columns(sec)
           my_conv={'Molecular Weight' : convert_dtype}
           filename = parser.check_get(sec,'activity_csv')
           self.df = pd.read_csv(filename,error_bad_lines=True, index_col=False, converters=my_conv)
           mapgn=parser.check_get(sec,'target_mapping')
           end_point_lst=parser.check_get(sec,"end_points").split(',')
           end_point_lst =  [ x.lower() for x in end_point_lst ]
           ##warning this assumes first column is the key and second column is the value
           chk = pd.read_csv(mapgn)
           chk=chk.dropna()
           self.target_dict={}
           for kv in raw_target_lst :
               res=chk[(chk[self.target_name_col] == kv.lower())]
               self.target_dict[kv]=res[self.target_id_col].values[0]
               
           ## make lower case to match previous mapping
           self.df[self.target_name_col] = self.df[self.target_name_col].str.lower()
           self.df[self.target_id_col] = self.df[self.target_name_col].map(self.target_dict)
           self.df=self.df.dropna( subset=[self.target_id_col] )
           self.df.rename( columns={"Molecule ChEMBL ID" : self.id_col, "Standard Relation" : self.relation_col }, inplace=True)
           self.df.rename( columns={"Standard Type" : "standard_type" }, inplace=True)

           self.df.standard_type = self.df.standard_type.str.lower()
           self.df=self.df[self.df.standard_type.isin( end_point_lst ) ]

       # when the gene target label isn't standardized, need to return the gene target mapping
    def filter_task(self, target_id):
        return self.df[(self.df[self.target_id_col] == self.target_dict[target_id])],self.target_dict[target_id]
