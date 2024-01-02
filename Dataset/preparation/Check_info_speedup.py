from pathlib import Path
import scipy.io as sio
import numpy as np
import os
from multiprocessing import Pool
from utils import load_dataset_from_txt


def poolfunc(mat_folder):
    # print(mat_folder)
    matdata1 = sio.loadmat(os.path.join(mat_folder, 'DTI_MD.mat'))
    matdata2 = sio.loadmat(os.path.join(mat_folder, 'DTI_Trace.mat'))
    matdata3 = sio.loadmat(os.path.join(mat_folder, 'Flair.mat'))
    matdata4 = sio.loadmat(os.path.join(mat_folder, 'Tabular_data_manualfeatures.mat'))

    DTI_MD = matdata1['DTI_MD']
    DTI_Trace = matdata2['DTI_Trace']
    Flair = matdata3['Flair']
    Tabular_data = np.squeeze(matdata4['Tabular_data'], axis=0)  # (105)

    patient_FRS_dict = matdata4['patient_Structure_Regs_dict']  # (12,12)
    node_feature = patient_FRS_dict['node_features'][0][0]

    invalid_ps = []
    networks = patient_FRS_dict['connection_map'][0][0] - np.eye(134)
    Ps = list(np.squeeze(np.where(np.sum(networks, axis=1) == 0), axis=0))
    if len(Ps):
        for p in Ps:
            invalid_ps.append(p)

    if len(np.argwhere(np.isnan(Tabular_data))):
        print('there are some nan value in ', mat_folder)

    channels_MD = np.mean(DTI_MD[:])
    channels_squared_MD = np.mean(DTI_MD ** 2, axis=(0, 1, 2))

    channels_Trace = np.mean(DTI_Trace[:])
    channels_squared_Trace = np.mean(DTI_Trace ** 2, axis=(0, 1, 2))

    channels_Flair = np.mean(Flair[:])
    channels_squared_Flair = np.mean(Flair ** 2, axis=(0, 1, 2))
    return channels_MD, channels_squared_MD, channels_Trace, channels_squared_Trace, channels_Flair, channels_squared_Flair, Tabular_data, node_feature, invalid_ps

#
pathes = ['../Data/strokenormdataset/DEMDAS_all_1.txt',
          '../Data/strokenormdataset/DEMDAS_all_0.txt']

mat_folders = []
for filepath in pathes:
    mat_folders += load_dataset_from_txt(filepath)

sample_num = mat_folders.__len__()

pool = Pool(16)
result = pool.map(poolfunc, mat_folders)
pool.close()
result = np.asarray(result, dtype=object)
invalid_plist = []
for m in list(result[:, 8]):
    invalid_plist = invalid_plist + m
invalid_plist = np.unique(np.asarray(invalid_plist))
All_node_data = np.concatenate([m[np.newaxis, :] for m in list(result[:, 7])], axis=0)
All_node_mean = np.mean(All_node_data, axis=(0,1))
All_node_std = np.std(All_node_data, axis=(0,1))

All_tabular_data = np.concatenate([m[np.newaxis, :] for m in list(result[:, 6])], axis=0)
All_tabular_mean = np.mean(All_tabular_data, axis=0)
All_tabular_std = np.std(All_tabular_data, axis=0)

dataset_DTI_MD_mean = np.mean(result[:, 0])
dataset_DTI_MD_std = (np.mean(result[:, 1]) - dataset_DTI_MD_mean ** 2) ** 0.5

dataset_DTI_Trace_mean = np.mean(result[:, 2])
dataset_DTI_Trace_std = (np.mean(result[:, 3]) - dataset_DTI_Trace_mean ** 2) ** 0.5

dataset_DTI_Flair_mean = np.mean(result[:, 4])
dataset_DTI_Flair_std = (np.mean(result[:, 5]) - dataset_DTI_Flair_mean ** 2) ** 0.5

print("invalid node list (too small/ or not exist in some samples)")
print(invalid_plist)

print(">>Dataset DEDEMAS and DEMDAS")
print("Mean MD: ", dataset_DTI_MD_mean)
print("Std MD: ", dataset_DTI_MD_std)

print("Mean Trace: ", dataset_DTI_Trace_mean)
print("Std Trace: ", dataset_DTI_Trace_std)

print("Mean Flair: ", dataset_DTI_Flair_mean)
print("Std Flair: ", dataset_DTI_Flair_std)

print("Mean tabular: ", All_tabular_mean)
print("Std tabular: ", All_tabular_std)

# print("for augmentation Mean tabular:", All_tabular_mean)
# print("for augmentation Std tabular: ", All_tabular_std)

norm_mat = {'Mean_tabular': All_tabular_mean,
            'Std_tabular':All_tabular_std,
            'Mean_node': All_node_mean,
            'Std_node': All_node_std,
            }

sio.savemat('../Data/strokenormdataset/norm_mfs_mat.mat', norm_mat)
