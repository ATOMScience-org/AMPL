import pandas as pd
import argparse

def parse_args():
    """
    Parse command-line arguments for comparing two split files.
    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
            - split1 (str): Path to the first split file.
            - split2 (str): Path to the second split file.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('split1', type=str, help='First split file')
    parser.add_argument('split2', type=str, help='Second split file')
    
    return parser.parse_args()

def compare_split_files(split1, split2):
    """
    Compare two CSV files containing dataset splits.

    This function reads two CSV files and compares their contents using the
    `compare_splits` function.

    Args:
        split1 (str): Path to the first CSV file.
        split2 (str): Path to the second CSV file.

    Returns:
        None: The function performs the comparison but ignores the return value from `compare_splits`.
    """
    compare_splits(pd.read_csv(split1), pd.read_csv(split2))

def compare_splits(split1, split2):
    """
    Compare two data splits to determine if they are effectively the same.
    This function checks whether two pandas DataFrames, `split1` and `split2`, 
    have the same subset/fold groupings and whether the compounds within each 
    group are identical. It performs the following checks:
    1. Verifies that the subset/fold pairs in both splits are identical.
    2. Ensures that the number of compounds in each subset/fold group is the same.
    3. Confirms that the compounds in each subset/fold group are identical.
    4. Checks that the sorted lists of compounds in each group are identical.
    If any of these checks fail, the function prints detailed information about 
    the differences and returns `False`. If all checks pass, it prints a success 
    message and returns `True`.
    Args:
        split1 (pd.DataFrame): The first data split to compare. Must contain 
            columns `subset`, `fold`, and `cmpd_id`.
        split2 (pd.DataFrame): The second data split to compare. Must contain 
            columns `subset`, `fold`, and `cmpd_id`.
    Returns:
        bool: `True` if the splits are effectively the same, `False` otherwise.
    Notes:
        - The function assumes that the input DataFrames are grouped by the 
          columns `subset` and `fold`.
        - If there are differences in the compounds, the function prints the 
          number of differing compounds and lists them if the count is less 
          than 10.
    """
    groups1 = split1.groupby(['subset', 'fold'])
    groups2 = split2.groupby(['subset', 'fold'])

    group1_names = set(groups1.groups.keys())
    group2_names = set(groups2.groups.keys())
    same_groups = group1_names == group2_names
    if not same_groups:
        print('There are different subset/fold pairs')
        print('Group1')
        print(group1_names)
        print('Group2')
        print(group2_names)
        return False

    # check that compounds in each group are the same
    for group_name in group1_names:
        g1 = groups1.get_group(group_name)
        g2 = groups2.get_group(group_name)

        if len(g1) != len(g2):
            print(f'Subset {group_name} is not the same size in both splits.')
            print(f'Group 1 has length {len(g1)} Group 2 has length {len(g2)}')
        
        gc1 = set(g1['cmpd_id'].values)
        gc2 = set(g2['cmpd_id'].values)

        if gc1 != gc2:
            print(f'Subset {group_name} contains different compounds.')
            ing1_notg2 = gc1-gc2
            ing2_notg1 = gc2-gc1
            print(f'There are {len(ing1_notg2)} compounds in split 1 and not split 2')
            if len(ing1_notg2)<10:
                print(ing1_notg2)
            print(f'There are {len(ing2_notg1)} compounds in split 2 and not split 1')
            if len(ing2_notg1)<10:
                print(ing2_notg1)
            return False

        gs1 = sorted(list(g1['cmpd_id'].values))
        gs2 = sorted(list(g2['cmpd_id'].values))

        if not all([gs1i==gs2i for gs1i, gs2i in zip(gs1, gs2)]):
            print('Sorting the list of compounds from both splits results in two different lists.')
            return False

    print('Both splits are effectively the same')
    return True

if __name__ == '__main__':
    args = parse_args()

    compare_split_files(args.split1, args.split2)