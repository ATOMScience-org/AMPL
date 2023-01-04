import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('split1', type=str, help='First split file')
    parser.add_argument('split2', type=str, help='Second split file')
    
    return parser.parse_args()

def compare_split_files(split1, split2):
    compare_splits(pd.read_csv(split1), pd.read_csv(split2))

def compare_splits(split1, split2):
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