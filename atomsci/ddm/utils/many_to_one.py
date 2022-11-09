import pandas as pd

def has_nans(df, col):
    total = len(df)
    after = len(df[col].dropna())
    return total!=after

def no_nan_ids_or_smiles(df, smiles_col, id_col):
    if has_nans(df, smiles_col):
        raise NANSMILESException('NANs found in SMILES column')
    
    if has_nans(df, id_col):
        raise NANCompoundIDException('NANs found in ID column')

    return True

def many_to_one(fn, smiles_col, id_col):
    df = pd.read_csv(fn)
    return many_to_one_df(df, smiles_col, id_col)

def many_to_one_df(df, smiles_col, id_col):
    '''
    AMPL requires that SMILES and compound_ids have a many to one mapping. 
    This function opens the dataset and checks this restraint. It will also
    check if any SMILES or compound_ids are empty/nan

    Arguments:
        df (pd.DataFrame): The DataFrame in question.
        smiles_col (str): The column containing SMILES.
        id_col (str): The column containing compound ids

    Returns:
        True if there is a many to one mapping. Raises one of 3 errors if it:
            - Has nan compound_ids
            - Has nan SMILES
            - Is not a many to one mapping between compound_ids and SMILES
    '''
    no_nan_ids_or_smiles(df, smiles_col, id_col)

    # if a compound id is associated with more than one SMILES
    id_one = df.drop_duplicates(subset=[smiles_col, id_col]).groupby(id_col)[smiles_col].count().max()
    if id_one > 1:
        raise ManyToOneException('SMILES and Compound IDs do not have a many to one mapping.')

    # SMILES can be associated with many compound ids no need to check them
    return True

class ManyToOneException(Exception):
    pass

class NANCompoundIDException(Exception):
    pass

class NANSMILESException(Exception):
    pass