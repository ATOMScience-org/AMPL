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

def one_to_one(fn, smiles_col, id_col):
    df = pd.read_csv(fn)
    return one_to_one_df(df, smiles_col, id_col)

def one_to_one_df(df, smiles_col, id_col):
    '''
    AMPL requires that SMILES and compound_ids have a one to one mapping. 
    This function opens the dataset and checks this restraint. It will also
    check if any SMILES or compound_ids are empty/nan

    Arguments:
        df (pd.DataFrame): The DataFrame in question.
        smiles_col (str): The column containing SMILES.
        id_col (str): The column containing compound ids

    Returns:
        True if there is a one to one mapping. Raises one of 3 errors if it:
            - Has nan compound_ids
            - Has nan SMILES
            - Is not a one to one mapping between SMILES and compound_ids
    '''
    no_nan_ids_or_smiles(df, smiles_col, id_col)

    # courtesy of https://stackoverflow.com/questions/50643386/easy-way-to-see-if-two-columns-are-one-to-one-in-pandas
    first = df.drop_duplicates([smiles_col, id_col]).groupby(smiles_col)[id_col].count().max()
    second = df.drop_duplicates([smiles_col, id_col]).groupby(id_col)[smiles_col].count().max()

    if first + second == 2:
        return
    else:
        raise OneToOneException('SMILES and Compound IDs do not have a one to one mapping.')

class OneToOneException(Exception):
    pass

class NANCompoundIDException(Exception):
    pass

class NANSMILESException(Exception):
    pass