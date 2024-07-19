import matplotlib
matplotlib.use('Agg')
import pandas as pd


#import custom_config
from atomsci.ddm.utils import struct_utils as su
from atomsci.ddm.utils import curate_data
import atomsci.ddm.pipeline.diversity_plots as dp
import atomsci.ddm.pipeline.chem_diversity as cd
from rdkit.Chem import Descriptors
from rdkit import Chem
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import argparse

import configparser

class CustomConfigParser(configparser.ConfigParser) :
   def __init__(self):
        super().__init__()
        self.def_sec="default"

   def check_get(self,section,keyval) :
      """Purpose is to check the default section for  a parameter value if its not found in the section"""

      try :
          rval = super().get(section,keyval)
      except configparser.NoOptionError :
          try :
              rval = super().get(self.def_sec,keyval)
          except :
              rval = None
      return rval

def parse_args():
    """Parse commandline arguments and return a Namespace.

    User must provide the configuration file to run this script
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-config_file', 
                default='priority_targets.ini',
                help='path to csv file containing config file')

    return parser.parse_args()

def save_combined_data(output_data_dir,output_img_dir, comb, comb_type="pre_curated") :
    """Write final curated datasets and associated diagnostic plots to file

    Args:
       output_data_dir (str) : directory location to put combined model ready dataset and rejected compounds.
       output_img_dir (str) : location to put diagnostic data , currently just distribution of activity values for final set"
       comb (dictionary) : dictionary with key as gene target and value as CustomActivityDump class
       comb_type (str): pre_curated (default) combines the datasets from different sources that were individually curated. Not yet implemented is a raw option to re-combine all data
    """
    for target_name in comb :
      print("process target",target_name)
      ofile=output_img_dir+target_name+"_"+comb_type+'_smiles_overlap.pdf'
      with PdfPages(ofile) as pdf :
         fig = plt.figure()
         comb_ds = CombineAMPLDataset( comb[target_name], comb_type )
         ld = comb_ds.lead_ds
         combine_df = comb_ds.combine_df

         sub_df,rejected_outliers=ld.combine_replicates(combine_df,True)
         ########################
         # Save rejected compounds to a file for future inspection
         ########################
         raw_ofile=output_data_dir+target_name+"_"+comb_type+'_rejected.csv'
         rejected_outliers.to_csv(raw_ofile,index=False)

         raw_ofile=output_data_dir+target_name+"_"+comb_type+'_total.csv'
         sub_df.to_csv(raw_ofile,index=False)

         sns.distplot(sub_df[ld.value_col],kde=False)
         label = "Distribution of combined activity values"
         plt.title(label)
         pdf.savefig(fig)

class ActivitySummary:
    """Class holds list of targets to be curated

    Currently this is responsible for just listing the targets.

    """

    def __init__(self, df):
        """Sets column names and reads the table."""
        self.df = df

        #this requires some naming convention
        self.target_id_col = 'Target Name'
        #self.target_name_col = 'Target Name'
        #self.standard_col = 'Standard Type'

    def make_filename_by_index(self, index):
        """Creates a file name with a unique task name that combines the HUGO target name and the
        measured paramter aka Standard Type

        Args:
            src: Path to table containing summary for the gpcr targets

        Returns:
            A string filename ending in the .csv extension
        """
        r = self.df.iloc[index]

        return '%s_%s.csv'%(r[self.target_name_col], r[self.standard_col])

    def iterrows(self):
        """Accessor function for self.sum_df.iterrows()

        Returns:
            iterator for the rows of the underlying table, self.sum_df
        """

        return self.df.iterrows()

    def get_target_name_lst(self) :
        return self.df[self.target_id_col].values

class AMPLDataset:
    """Manage access to bioactivty datasets

    """
    def __init__(self) :
        self.df =None
        self.smiles_col = None
        self.base_smiles_col = None
        self.id_col = None
        self.standard_col = None
        self.target_name_col = None
        self.target_id_col = None
        self.relation_col = None
        self.value_col = None
        self.date_col = None

    def get_smiles(self):
        """Return the list of SMILES strings for this dataset

        Returns:
            list: The list of SMILES strings
        """
        return self.df[self.smiles_col].values

    def filter_task(self, target_id):
        """Return a dataframe for a userspecified target name

        Args:
            target_id (str): target name to select from dataset with possibly multiple targets

        Returns:
            pandas dataframe: subset of the bioactivity data
            str: returns input target_id value

        """
        return self.df[(self.df[self.target_id_col] == target_id)],target_id

    def add_base_smiles_col(self):
        """Calculate base rdkit smiles and add them to the dataframe

           Uses AMPL's atomsci.ddm.utils.struct_utils.base_smiles_from_smiles procedure to
           create a canonicalized form of the SMILES input
        """
        self.df[self.base_smiles_col] = self.df[self.smiles_col].apply(su.base_smiles_from_smiles,workers=16)

    def drop_na_values_base_smiles(self):
        """Remove rows in dataframe where the canonicalized SMILES string is empty

        Called after the add_base_smiles_col() function.  Some raw input SMILES may fail
        to succesfully canonicilize using the current procedure.

        Returns:
           pandas dataframe: dataframe with dropped rows
        """
        self.df.replace('', np.nan, inplace=True)
        new_df = self.df.dropna(subset=[self.base_smiles_col])
        new_df = new_df.dropna(subset=[self.value_col])
        dropped_rows = self.df[~(self.df.index.isin(new_df.index))]
        self.df = new_df
        return dropped_rows

    def filter_properties(self,parser,sec) :
        """Remove rows in dataframe based on user defined filtering properties

        Currently two filtering properties are supported mol_weight and p_activity (-log activity value)

        Args:
            parser (configparser) : holds values of the filtering parameters
            sec (str) : specifies whether filtering is specified for a specific data source

        Returns:
            returns a dataframe with the rows that failed to pass the filtering criteria

        Examples:
            Set the following parameters in the configuration file
            filter on molecular weight outside the range of 0 to 2000 inclusive
            mol_weight = 0:2000

            filter on log10 nm activity outside the range 2 to 14 inclusive
            p_activity = 2:14
        """
        mn,mx=parser.check_get(sec,'mol_weight').split(':')
        self.df['tmp_col1'] = self.df[self.base_smiles_col].apply(Chem.MolFromSmiles)
        mw_lst=[]
        for idx,row in self.df.iterrows() :
            sml=row[self.base_smiles_col]
            mol=row['tmp_col1']
            mw = Descriptors.MolWt(mol)
            mw_lst.append(mw)
        self.df.insert(0,'tmp_col2',mw_lst) #self.df['tmp_col1'].apply(Descriptors.MolWt,workers=16)
        self.df = self.df[ (self.df['tmp_col2'] >= float(mn)) & (self.df['tmp_col2'] <= float(mx)) ]
        rej1=self.df[ (self.df['tmp_col2'] < float(mn)) | (self.df['tmp_col2'] > float(mx)) ]
        self.df.drop(columns=['tmp_col1','tmp_col2'],axis=1,inplace=True)

        mn,mx=parser.check_get(sec,'p_activity').split(':')
        self.df = self.df[ (self.df[self.value_col] >= float(mn)) & (self.df[self.value_col] <= float(mx)) ]
        rej2=self.df[ (self.df[self.value_col] < float(mn)) | (self.df[self.value_col] > float(mx)) ]
        rej= pd.concat([rej1,rej2])
        if not rej.empty :
            print("some molecule failed property filters")
            print(rej)
        return rej

    def combine_replicates(self,data_frame,ignore_compound_id, tolerance=10,max_std=1,output_value_col=None, label_actives=True, active_thresh=None,date_col=None) :
        """Combine replicates by taking average and discarding molecules with high variation in the measured value

        Args:
            data_frame: target specific subset of data_frame
            ignore_compound_id: when combing replicates across data sources, we have to assume the same compound will have differnt IDs, so you must use the smiles to match across datasources.
            tolerance: percent variation between replicates tolerated
            max_std: maximum standard deviation between replicates tolerated
            output_value_col: Optional; the column name to use in the output data frame for the averaged data.
            label_actives: If True, generate an additional column 'active' indicating whether the mean value is above a threshold specified by active_thresh.
            active_thresh: The threshold to be used for labeling compounds as active or inactive.
                       If active_thresh is None (the default), the threshold used is the minimum reported value across all records
                       with left-censored values (i.e., those with '<' in the relation column).
            date_col: The input data frame column containing dates when the assay data was uploaded. If not None, the code will assign the earliest
                 date among replicates to the aggregate data record.
        Returns
            A data frame of compounds with averaged values and a dataframe with compounds that were rejected as having too much variation
        """

        column = self.value_col
        smiles_col = self.base_smiles_col
        compound_id = self.id_col
        if ignore_compound_id :
            compound_id = smiles_col
        relation_col = self.relation_col
        ##################################################
        ### This is to run a diagnostic to look for and report outlieres
        ### Outliers are then removed 
        ### TODO: we might want to make option to report outliers without removing them
        ##################################################
        ### list_bad_duplicates set to Yes would be redundant since they are saved to file now
        list_bad_duplicates='No'
        curated_df = curate_data.average_and_remove_duplicates(column, tolerance, 
                                                                list_bad_duplicates, 
                                                                data_frame, max_std, 
                                                                compound_id=compound_id, 
                                                       smiles_col=smiles_col)
        save_ids=curated_df[compound_id].unique().tolist()
        reject=data_frame[~(data_frame[compound_id].isin(save_ids))]
        keep_df=data_frame[data_frame[compound_id].isin(save_ids)]
        
        ### 
        ### TODO: THIS HARDCODES THE COLUMN NAMES AND SHOULD USE id_col value to set the column name
        ###       for now the "compound id" needs to be called "compound_id" 
        ###       "relation" needs to be called "relation" !!!!! 
        ###        Need to change this
        data_frame=curate_data.aggregate_assay_data(keep_df, value_col=column, output_value_col=None,
                             label_actives=True,
                             active_thresh=None,
                             id_col=compound_id, smiles_col=smiles_col, relation_col=relation_col)

        return data_frame,reject 

class CombineAMPLDataset:
    """Class responsibe for combining data from multiple sources into single data frame"""
    def __init__(self, ds_lst, input_dtype):
        """Enumerate through multiple datasets taken from multiple sources, use first datasource to set the column header definitions
        and concatenate

        Args:
            ds_lst: list of AMPLDataset objects to be combined into single data frame
            input_dtype: pre_curated or raw  ; if data is raw it means we will be running the de duplication functions on the
                            original data, not the individually curated data sources
        
        TODO:  'raw' is not implemented yet!! See below
        """

        """
        TODO:
        if input_dtype == 'raw' : 
            run deduplication procedure that removes duplicates from different sources
            do this by treating any pair of matching SMILES  with near exact log difference (e.g. <0.1) as being the same measurement; this will ensure that we're not double counting
        """

        lead_ds=ds_lst[0]
        self.lead_ds = lead_ds
        df_lst=[]
        ## normalize column names
        save_cols=[lead_ds.id_col, lead_ds.value_col, lead_ds.base_smiles_col , lead_ds.relation_col ]
        for it in range(1,len(ds_lst),1) :
            ds=ds_lst[it]
            orig_cols=[ds.id_col, ds.value_col, ds.base_smiles_col , ds.relation_col ]
            rl_df=ds.df.rename( columns={ ds.id_col: lead_ds.id_col, ds.value_col : lead_ds.value_col, ds.base_smiles_col : lead_ds.base_smiles_col, ds.relation_col : lead_ds.relation_col, ds.smiles_col : lead_ds.smiles_col   }, inplace=False)
            rl_df=rl_df[ save_cols ] 
            df_lst.append(rl_df)
        ## add lead dataframe to list
        rl_df=lead_ds.df[ save_cols ] 
        df_lst.append(rl_df)
        self.combine_df = pd.concat(df_lst)
         
                      
if __name__ == '__main__':
    args = parse_args()

    #save all data for each target here from each source
    # save curated form and raw form to combine later using either option
    comb,raw_comb={},{}
    print("read config file",args.config_file)
    # SR
    #parser = custom_config.CustomConfigParser()
    parser = CustomConfigParser()
    parser.read(args.config_file)
    ## initiate global/default settings
    def_sec = 'default'
    for sec in parser.sections() :
      print("section",sec)
      if sec != def_sec :
         module_name = parser.check_get(sec,'parse_module')
         class_name = parser.check_get(sec,'parse_module_class')
         config_plot = parser.check_get(sec, 'plot') == "True"
         print("check",module_name,class_name)
         module = __import__(module_name)

         class_name = parser.check_get(sec,'parse_module_class')
         config_args = parser.check_get(sec,'activity_summary')
         act_sum = ActivitySummary(pd.read_csv(config_args))

         CustomActivityDump = getattr(module, class_name)
         act_data = CustomActivityDump(parser=parser,sec=sec,raw_target_lst=act_sum.get_target_name_lst())
         data_source_name=act_data.data_source_name

         #######################################
         # Generate canonical base SMILES strings
         #######################################
         act_data.add_base_smiles_col()

         reject_properties=act_data.filter_properties(parser,sec)

         ########################
         # Clean up the data
         ########################
         dropped_rows = act_data.drop_na_values_base_smiles()
         config_args = parser.check_get(sec,'output_data_dir')
         raw_ofile=config_args+data_source_name+'_dropped_raw_smiles.csv'
         dropped_rows.to_csv(raw_ofile,index=False)
         ## TODO: should enforce unique list of target names (no duplicates) but currently doesn't
         ## enumerate and curate each target separately
         for i, r in act_sum.iterrows():
            raw_target_name=r[act_sum.target_id_col]
            raw_sub_df,target_name = act_data.filter_task(raw_target_name)
            print(raw_target_name,target_name)
            if raw_sub_df.empty :
               print("WARNING no data found for",raw_target_name,target_name, sec)
               continue
            ########################
            # Save original raw data by target to LC
            ########################
            #########################
            ## There should be a datastore equivalent here to checkpoint the raw dataset in the datastore
            #########################
            output_data_dir = parser.check_get(sec,'output_data_dir')
            raw_ofile=output_data_dir+target_name+'_'+data_source_name+'_raw_smiles.csv'
            raw_sub_df.to_csv(raw_ofile,index=False)

            ########################
            # Save rejected compounds to a file for future inspection
            ########################
            sub_df,rejected_outliers=act_data.combine_replicates(raw_sub_df,False)
            all_rej = pd.concat([rejected_outliers,reject_properties])
            raw_ofile=output_data_dir+target_name+'_'+data_source_name+'_rejected.csv'
            all_rej.to_csv(raw_ofile,index=False)

            #########################
            ## There should be a datastore equivalent here to checkpoint the curated dataset in the datastore
            #########################
            cur_ofile=output_data_dir+target_name+'_'+data_source_name+'_cur_smiles.csv'
            sub_df.to_csv(cur_ofile,index=False)

            ## don't try to draw heatmaps or histograms if datset has fewer than 10 compounds
            if sub_df.shape[0] >= 10 and config_plot:
               #########################
               ## Save image diagnostics to pdf
               #########################
               print("Info>> Make diagnostics images =====================")
               output_img_dir = parser.check_get(sec,'output_img_dir')
               ofile=output_img_dir+target_name+'_'+data_source_name+'_raw_smiles.pdf'
               with PdfPages(ofile) as pdf :
                 fig = plt.figure()
                 sns.distplot(sub_df[act_data.value_col],kde=False)
                 label = "Distribution of activity values"
                 plt.title(label)
                 pdf.savefig(fig)
                 #plt.close()
                 plt.clf()

                 #########################
                 ## Don't try to cluster if the dataset is too large, to avoid excessive runtimes 
                 #########################
                 if sub_df.shape[0] < int(parser.check_get(sec,'max_clus_size')) :
                    print("passed",sub_df.shape,)
                    dp.diversity_plots(dset_key=cur_ofile,datastore=False,id_col=act_data.id_col,smiles_col=act_data.base_smiles_col, is_base_smiles=True,response_col=act_data.value_col,max_for_mcs=100,out_dir=output_img_dir)

                 feat_type='ECFP'
                 dist_metric='tanimoto'
                 smiles_lst1=sub_df[act_data.base_smiles_col].tolist()              
                 calc_type='nearest'
                 dist_sample=cd.calc_dist_smiles(feat_type,dist_metric,smiles_lst1,None,calc_type)
                 sns.distplot(dist_sample,kde=False,axlabel=feat_type+'_'+dist_metric+'_'+calc_type)
                 label = "Nearest distance between compound pairs"
                 plt.title(label)
                 pdf.savefig(fig)

            ## save curated form of each dataset
            if target_name not in comb :
               comb[target_name]=[]
            ndata = CustomActivityDump(dataset=act_data,df=sub_df)
            comb[target_name].append(ndata)

            ## save raw form gives option to combine data from all sources together
            if target_name not in raw_comb :
               raw_comb[target_name]=[]
            raw_ndata = CustomActivityDump(dataset=act_data,df=raw_sub_df)
            raw_comb[target_name].append(ndata)
   
    #assumes def_sec has an output image directory specification
    output_img_dir = parser.check_get(def_sec,'output_img_dir')
    output_data_dir = parser.check_get(def_sec,'output_data_dir')
    save_combined_data(output_data_dir,output_img_dir,comb,comb_type="pre_combined")
    # raw option is not yet implemented
    #save_combined_data(output_data_dir,output_img_dir,raw_comb,dtype="raw")
