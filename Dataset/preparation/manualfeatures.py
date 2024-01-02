# -*- coding:utf-8 -*-
"""
@Time: 2023/1/14 21:34
@Author: Shuting Liu & Baochang Zhang
@IDE: PyCharm
@File: fuck.py
@Comment: #Enter some comments at here
"""
from pathlib import Path
import glob
import pandas as pd
from tqdm import tqdm
import os
import scipy.io as sio
import numpy as np



# # check
# matdata = sio.loadmat('/home/shuting/Datasets/DEMDAS_preproc_liu_norm/P_DD_BBF_2IA8Z4EU/Tabular_data_manualfeatures.mat')
# Tabular_data_missing = matdata['Tabular_data_missing']  # (12,12)
# Tabular_data = matdata['Tabular_data']  # (12,12)
# print('Tabular_data_missing: ',  np.shape(Tabular_data_missing), Tabular_data_missing)
# print('Tabular_data: ',  np.shape(Tabular_data), Tabular_data)
# patient_FRS_dict = matdata['patient_Structure_Regs_dict']  # (12,12)
# networks = patient_FRS_dict['connection_map'][0][0]-np.eye(134)
# node_position = patient_FRS_dict['roi_Centroid'][0][0]
# node_feature = patient_FRS_dict['node_features'][0][0]
# node_id = patient_FRS_dict['label_list'][0][0][0]
# print('node_feature: ', np.shape(node_feature), node_feature)
#
# exit(0)


# sheet_name = 'DEMDAS'
# dataset_root = '/home/shuting/Datasets/'+sheet_name+'_preproc_liu_norm'
# file_list = glob.glob(str(Path(dataset_root)/'*/Tabular_data.mat'))

excelfile = '../Data/Tabular_data.xlsx'
sheet_name = 'DEMDAS'
tabledata = pd.read_excel(excelfile, sheet_name=sheet_name)
ID_manualfeatures = ['ID',
                     'SVDscore_total',
                     'lacune_count',
                     'PVS_level',
                     'Fazekas_PVWM',
                     'Fazekas_DWM',
                     'CMB_count']
tabledata = tabledata[ID_manualfeatures]
tabledata = tabledata.dropna()
IDs = tabledata['ID'].to_list()
Mfsdata = tabledata[ID_manualfeatures[1:]].to_numpy()
Mfsdata_missing = tabledata[ID_manualfeatures[1:]].isnull().to_numpy().astype('int')
num = 0
for index, id in tqdm(enumerate(IDs), total=IDs.__len__()):
    file_check = '/home/shuting/Datasets/DEMDAS_preproc_liu_norm/'+id+'/Tabular_data.mat'
    if os.path.exists(file_check):
        print(file_check)
        manualfeatures = Mfsdata[index,:]

        manualfeatures[0] = manualfeatures[0] / 4.0
        manualfeatures[1] = manualfeatures[1] / 12.0
        manualfeatures[2] = manualfeatures[2] / 3.0
        manualfeatures[3] = manualfeatures[3] / 3.0
        manualfeatures[4] = manualfeatures[4] / 26.0

        missing_v = Mfsdata_missing[index,:]
        matdata = sio.loadmat(file_check)
        Tabular_data = np.squeeze(matdata['Tabular_data'])
        Tabular_data_missing = np.squeeze(matdata['Tabular_data_missing'])
        Tabular_data = np.append(Tabular_data, manualfeatures)
        Tabular_data_missing = np.append(Tabular_data_missing,missing_v)
        Tabular_data[0]=Tabular_data[0]/100.0
        Tabular_data[2] = Tabular_data[2] / 12.0-1.0

        new_file = '/home/shuting/Datasets/DEMDAS_preproc_liu_norm/'+id+'/Tabular_data_manualfeatures.mat'

        mat_tabuler_dict = {"Tabular_data": Tabular_data.astype(np.float32),
                            "Tabular_data_missing": Tabular_data_missing.astype(np.int32),
                            "patient_Structure_Regs_dict": matdata['patient_Structure_Regs_dict']}

        sio.savemat(new_file, mat_tabuler_dict)