# -*- coding: utf-8 -*-
"""
数据集构建工具

用于构建和划分数据集，生成 JSON 格式的数据集描述文件
"""

import os
import json


def build_json_file():
    """
    构建数据集划分文件

    根据原始划分文件和实际存在的数据文件，
    生成包含有效数据的划分文件

    输出文件: present_dataset_split.json
        包含 train, val, test 三个子集
        每个子集包含文件路径和标签
    """
    # 读取原始划分文件
    json_path = '/mnt/4T/chenhj/datasets/my_dataset/new_dataset/dataset_split.json'
    with open(json_path, 'r') as f:
        original_split = json.load(f)

    # 初始化原始划分字典
    original_split_dict = {
        'train0': [],  # 训练集负样本
        'train1': [],  # 训练集正样本
        'val0': [],    # 验证集负样本
        'val1': [],    # 验证集正样本
        'test0': [],   # 测试集负样本
        'test1': []    # 测试集正样本
    }

    # 解析原始划分
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

    # 读取实际存在的数据文件
    data_path = '/mnt/4T/chenhj/datasets/my_dataset/new_dataset/extraction'
    data_category_list = os.listdir(data_path)

    # 初始化实际数据字典
    present_data_dict = {
        'train': [],
        'val': [],
        'test': []
    }

    # 遍历所有类别和文件
    for category in data_category_list:
        category_path = os.path.join(data_path, category)

        for file_name in os.listdir(category_path):
            # 获取不带扩展名的文件名
            name = f'{category}/{file_name}'

            # 检查是否属于训练集
            if (name[:-6] in original_split_dict['train0'] or
                name[:-7] in original_split_dict['train0']):
                present_data_dict['train'].append([name[:-4], 0])

            elif (name[:-6] in original_split_dict['train1'] or
                  name[:-7] in original_split_dict['train1']):
                present_data_dict['train'].append([name[:-4], 1])

            # 检查是否属于验证集
            elif (name[:-6] in original_split_dict['val0'] or
                  name[:-7] in original_split_dict['val0']):
                present_data_dict['val'].append([name[:-4], 0])

            elif (name[:-6] in original_split_dict['val1'] or
                  name[:-7] in original_split_dict['val1']):
                present_data_dict['val'].append([name[:-4], 1])

            # 检查是否属于测试集
            elif (name[:-6] in original_split_dict['test0'] or
                  name[:-7] in original_split_dict['test0']):
                present_data_dict['test'].append([name[:-4], 0])

            elif (name[:-6] in original_split_dict['test1'] or
                  name[:-7] in original_split_dict['test1']):
                present_data_dict['test'].append([name[:-4], 1])

    # 统计文件数量
    total_files = sum(len(files) for files in present_data_dict.values())
    print(f"共找到 {total_files} 个有效数据文件")
    print(f"  训练集: {len(present_data_dict['train'])} 个")
    print(f"  验证集: {len(present_data_dict['val'])} 个")
    print(f"  测试集: {len(present_data_dict['test'])} 个")

    # 保存为 JSON 文件
    output_path = '/mnt/4T/chenhj/datasets/my_dataset/new_dataset/present_dataset_split.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(present_data_dict, f, indent=4)

    print(f"数据集划分文件已保存: {output_path}")


def main():
    """主函数"""
    build_json_file()


if __name__ == '__main__':
    main()
