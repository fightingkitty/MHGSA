# -*- coding:utf-8 -*-
"""
@Time: 2022/10/28 12:18
@Author: Shuting Liu & Baochang Zhang
@IDE: PyCharm
@File: Download_dataset.py
@Comment: #Enter some comments at here
"""
import glob
from pathlib import Path
import os
import shutil

sheet_name = 'DEMDAS'
dataset_root = '/u/home/lius/Dataset/stroke/'+sheet_name+'/'+sheet_name+'_preproc_liu'
label_list = glob.glob(str(Path(dataset_root)/'*/label.mat'))
Tabular_list = glob.glob(str(Path(dataset_root)/'*/Tabular_data.mat'))

for file in label_list:
    new_file = file.replace(sheet_name+'_preproc_liu', sheet_name+'_preproc_noimage_liu')
    if os.path.exists(str(Path(new_file).parent)):
        shutil.rmtree(str(Path(new_file).parent))
        os.makedirs(str(Path(new_file).parent))
    else:
        os.makedirs(str(Path(new_file).parent))
    shutil.copyfile(file, new_file)

for file in Tabular_list:
    new_file = file.replace(sheet_name+'_preproc_liu', sheet_name+'_preproc_noimage_liu')
    shutil.copyfile(file, new_file)



