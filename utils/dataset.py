# -*-coding: Utf-8 -*-
# Author: CHJ
# Time: 2024/7/21 下午4:50

"""
This file is used to build the dataset.

-------------------------------------------
First, we need to produce the txt file.
"""

import os
import json

def build_json_file():
    """

    """

    # read json file
    with open('/mnt/4T/chenhj/datasets/my_dataset/new_dataset/dataset_split.json', 'r') as f:
        original_split = json.load(f)

    original_split_dict = {
        'train0': [],
        'train1': [],
        'val0': [],
        'val1': [],
        'test0': [],
        'test1': []
    }

    for d in original_split['train']:
        if d[1] == 0:
            original_split_dict['train0'].append(d[0])
        else:
            original_split_dict['train1'].append(d[0])
    for d in original_split['val']:
        if d[1] == 0:
            original_split_dict['val0'].append(d[0])
        else:
            original_split_dict['val1'].append(d[0])
    for d in original_split['test']:
        if d[1] == 0:
            original_split_dict['test0'].append(d[0])
        else:
            original_split_dict['test1'].append(d[0])

    # read the present data file list
    data_path = '/mnt/4T/chenhj/datasets/my_dataset/new_dataset/extraction'
    data_category_list = os.listdir(data_path)

    present_data_dict = {
        'train': [],
        'val': [],
        'test': [],
    }
    for category in data_category_list:
        for file_name in os.listdir(os.path.join(data_path, category)):
            name = f'{category}/{file_name}'
            if name[:-6] in original_split_dict['train0'] or name[:-7] in original_split_dict['train0']:
                present_data_dict['train'].append([name[:-4], 0])
            elif name[:-6] in original_split_dict['train1'] or name[:-7] in original_split_dict['train1']:
                present_data_dict['train'].append([name[:-4], 1])
            elif name[:-6] in original_split_dict['val0'] or name[:-7] in original_split_dict['val0']:
                present_data_dict['val'].append([name[:-4], 0])
            elif name[:-6] in original_split_dict['val1'] or name[:-7] in original_split_dict['val1']:
                present_data_dict['val'].append([name[:-4], 1])
            elif name[:-6] in original_split_dict['test0'] or name[:-7] in original_split_dict['test0']:
                present_data_dict['test'].append([name[:-4], 0])
            elif name[:-6] in original_split_dict['test1'] or name[:-7] in original_split_dict['test1']:
                present_data_dict['test'].append([name[:-4], 1])

    # write the present data file list
    num = 0
    for i in data_category_list:
        num += len(os.listdir(os.path.join(data_path, i)))

    with open('/mnt/4T/chenhj/datasets/my_dataset/new_dataset/present_dataset_split.json', 'w') as f:
        json.dump(present_data_dict, f, indent=4)

def main():
    build_json_file()

if __name__ == '__main__':
    main()