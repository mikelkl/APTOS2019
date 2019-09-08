#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time: 12/8/20196:04 PM
#@Author: AnguliaYang
#@File : cv_data_split.py

import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
import os
import numpy as np

save_path = '../input/atpos-data-split/'
root_dir_2015 = os.path.join('..//input/diabetic-retinopathy-resized/')
root_dir_2019 = os.path.join('../input/aptos2019-blindness-detection/')
train_2015 = '../input/diabetic-retinopathy-resized/resized_train_cropped/resized_train_cropped/'
train_2019 = '../input/aptos2019-blindness-detection/train_images/'
test_2019 = '../input/aptos2019-blindness-detection/test_images/'
n_splits = 5

df_2015 = pd.read_csv(root_dir_2015+'trainLabels_cropped.csv')
df_2015['path'] = df_2015['image'].map(lambda x: os.path.join(train_2015, '{}.jpeg'.format(x)))
skf1 = StratifiedKFold(n_splits=n_splits)

print('Processing 2015 data...')
X15 = df_2015['image']
y15 = df_2015['level']
c1=0
for train_idx1, val_idx1 in skf1.split(X15,y15):
    is_valid_2015 = np.zeros(len(df_2015), dtype=bool)
    print(len(val_idx1),val_idx1)
    c1+=1
    is_valid_2015[val_idx1] = True
    df_2015['is_valid'+str(c1)] = is_valid_2015

df_2015.to_csv(save_path+'df_2015_cv.csv')

df_2019 = pd.read_csv(root_dir_2019+'train.csv')
df_2019['path'] = df_2019['id_code'].map(lambda x: os.path.join(train_2019, '{}.png'.format(x)))
skf2 = StratifiedKFold(n_splits=n_splits)

print('Processing 2019 data...')
X19 = df_2019['id_code']
y19 = df_2019['diagnosis']
c2=0
for train_idx2,val_idx2 in skf2.split(X19,y19):
    is_valid_2019 = np.zeros(len(df_2019), dtype=bool)
    print(len(val_idx2), val_idx2)
    c2+=1
    is_valid_2019[val_idx2] = True
    df_2019['is_valid'+str(c2)] = is_valid_2019
df_2019.to_csv(save_path+'df_2019_cv.csv')

