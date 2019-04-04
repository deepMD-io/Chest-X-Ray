import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split

fn_in_train = '../data/CheXpert-v1.0-small/train.csv'
fn_in_valid = '../data/CheXpert-v1.0-small/valid.csv'

fn_out_train = '../data/CheXpert-v1.0-small/data_train.csv'
fn_out_valid = '../data/CheXpert-v1.0-small/data_valid.csv'
fn_out_test = '../data/CheXpert-v1.0-small/data_test.csv'

# read data and merge
df_train = pd.read_csv(fn_in_train)
df_valid = pd.read_csv(fn_in_valid)
df = pd.concat([df_train,df_valid])
print('Records in total:')
print(len(df))

# fillna(0) for 14 types
vars = df.columns.values
vars = vars[5:]
df[vars] = df[vars].fillna(0)

# find the patients id
ids = df['Path'].copy().values
for i, id in enumerate(ids):
	ids[i] = id[33:38]
df['id'] = ids

# unique patients' id
ids_unique = np.unique(ids)
# 64740 patients in total
print('Patiens in total:')
print(len(ids_unique))

# train, valid, test:  0.8, 0.1, 0.1
id_train, id_test = train_test_split(ids_unique, test_size=0.2, random_state = 2019)
id_valid, id_test = train_test_split(id_test, test_size=0.5, random_state = 403)
print('Patiens in train:')
print(len(id_train))
print('Patiens in valid:')
print(len(id_valid))
print('Patiens in test:')
print(len(id_test))

data_train = df[df['id'].isin(id_train)].drop(columns=['id'])
data_valid = df[df['id'].isin(id_valid)].drop(columns=['id'])
data_test = df[df['id'].isin(id_test)].drop(columns=['id'])

data_train.to_csv(fn_out_train, index = False)
data_valid.to_csv(fn_out_valid, index = False)
data_test.to_csv(fn_out_test, index = False)

print('Check records in total:')
print(len(data_train) + len(data_valid) + len(data_test))
