import numpy as np
import pandas as pd
import json
data=pd.read_csv(r'/mnt/sdb/home/hjy/Summary-DTA/data/KIBA/KIBA.csv')
# 假设您的DataFrame名为df，列名为'col1'，'col2'，'col3'
# 创建一个空的嵌套字典
nested_dict = {}

# 迭代DataFrame的每一行
for index, row in data.iterrows():
    key1 = str(row['COMPOUND_ID'])
    key2 = row['PROTEIN_ID']
    value = row['REG_LABEL']

    # 检查第一个键是否已存在于嵌套字典中
    if key1 in nested_dict:
        # 检查第二个键是否已存在于第一个键的字典中
        if key2 in nested_dict[key1]:
            # 如果第二个键已存在，则将内容添加到现有列表中
            nested_dict[key1][key2]= [value]
        else:
            # 如果第二个键不存在，则创建一个新的列表，并将内容添加到列表中
            nested_dict[key1][key2] = [value]
    else:
        # 如果第一个键不存在，则创建一个新的字典，并添加第二个键和内容
        nested_dict[key1] = {key2: [value]}
file_name=r'/mnt/sdb/home/hjy/Summary-DTA/data/KIBA/label.json'
with open(file_name, 'w') as file:
    json.dump(nested_dict, file)
'''
for fold in range(1, 6):
    print(fold)
    train_dir='data/Davis/processed/train/fold'+str(fold)
    test_dir='data/Davis/processed/test/fold'+str(fold)
    protein_id=np.load(train_dir+'/protein_id.npy')
    compound_id=np.load(train_dir+'/compound_id.npy')
    rows = []
    # 嵌套循环迭代所有可能的组合
    for value1 in compound_id:
        for value2 in protein_id:
            # 创建一个字典，包含当前组合的值
            row = {'compound_id': value1, 'protein_id': value2}
            rows.append(row)
    # 打印DataFrame
    df = pd.DataFrame(rows)
    df.to_csv(train_dir+'/idx_id.csv')
    protein_id = np.load(test_dir + '/protein_id.npy')
    compound_id = np.load(test_dir + '/compound_id.npy')
    rows = []
    # 嵌套循环迭代所有可能的组合
    for value1 in compound_id:
        for value2 in protein_id:
            # 创建一个字典，包含当前组合的值
            row = {'compound_id': value1, 'protein_id': value2}
            rows.append(row)
    # 打印DataFrame
    df = pd.DataFrame(rows)
    df.to_csv(test_dir + '/idx_id.csv')

'''


