import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from plot3dnextwork import show_graph_with_labels
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from preparation.utils import load_dataset_from_txt
from pathlib import Path
np.set_printoptions(suppress=True)
import time

pathes = ['../Data/strokenormdataset/DEMDAS_all_1.txt',
          '../Data/strokenormdataset/DEMDAS_all_0.txt']
mat_folders = []
for filepath in pathes:
    mat_folders += load_dataset_from_txt(filepath)

filepath = mat_folders[1]
t = time.time()
path1 = str(Path(filepath)/'DTI_MD.mat')
path2 = str(Path(filepath)/'DTI_Trace.mat')
path3 = str(Path(filepath)/'Flair.mat')
path4 = str(Path(filepath)/'infarct_ROI.mat')
path5 = str(Path(filepath)/'WMH_ROI.mat')
path6 = str(Path(filepath)/'DTI_MD_d2.mat')
path7 = str(Path(filepath)/'DTI_Trace_d2.mat')
path8 = str(Path(filepath)/'Flair_d2.mat')
path9 = str(Path(filepath)/'infarct_ROI_d2.mat')
path10 = str(Path(filepath)/'WMH_ROI_d2.mat')
path11 = str(Path(filepath)/'Tabular_data.mat')
path12 = str(Path(filepath)/'label.mat')

matdata1 = sio.loadmat(path1)
matdata2 = sio.loadmat(path2)
matdata3 = sio.loadmat(path3)
matdata4 = sio.loadmat(path4)
matdata5 = sio.loadmat(path5)
matdata6 = sio.loadmat(path6)
matdata7 = sio.loadmat(path7)
matdata8 = sio.loadmat(path8)
matdata9 = sio.loadmat(path9)
matdata10 = sio.loadmat(path10)
matdata11 = sio.loadmat(path11)
matdata12 = sio.loadmat(path12)

# exit(0)

DTI_MD = matdata1['DTI_MD']  # (80,95,77)
DTI_Trace = matdata2['DTI_Trace']  # (80,95,77)
Flair = matdata3['Flair']  # (80,95,77)
infarct_ROI = matdata4['infarct_ROI']  # (80,95,77)
WMH_ROI = matdata5['WMH_ROI']  # (80,95,77)

DTI_MD = (DTI_MD - np.min(DTI_MD[:])) / (np.max(DTI_MD[:]) - np.min(DTI_MD[:])) * 255
Flair = (Flair - np.min(Flair[:])) / (np.max(Flair[:]) - np.min(Flair[:])) * 255
DTI_Trace = (DTI_Trace - np.min(DTI_Trace[:])) / (np.max(DTI_Trace[:]) - np.min(DTI_Trace[:])) * 255

DTI_MD_d2 = matdata6['DTI_MD']  # (80,95,77)
DTI_Trace_d2 = matdata7['DTI_Trace']  # (80,95,77)
Flair_d2 = matdata8['Flair'] # (80,95,77)
infarct_ROI_d2 = matdata9['infarct_ROI']  # (80,95,77)
WMH_ROI_d2 = matdata10['WMH_ROI'] # (80,95,77)


DTI_MD_d2 = (DTI_MD_d2 - np.min(DTI_MD_d2[:])) / (np.max(DTI_MD_d2[:]) - np.min(DTI_MD_d2[:])) * 255
Flair_d2 = (Flair_d2 - np.min(Flair_d2[:])) / (np.max(Flair_d2[:]) - np.min(Flair_d2[:])) * 255
DTI_Trace_d2 = (DTI_Trace_d2 - np.min(DTI_Trace_d2[:])) / (np.max(DTI_Trace_d2[:]) - np.min(DTI_Trace_d2[:])) * 255

Tabular_data_missing = matdata11['Tabular_data_missing']  # (12,12)
Tabular_data = matdata11['Tabular_data']  # (12,12)
binary_label = matdata12['binary_label']  # (12,12)

# max_slice1 = np.argmax(np.sum(infarct_ROI, axis=(1, 2)))
# max_slice2 = np.argmax(np.sum(WMH_ROI, axis=(1, 2)))


print('Tabular_data_missing: ',  np.shape(Tabular_data_missing), Tabular_data_missing)
print('Tabular_data: ',  np.shape(Tabular_data), Tabular_data)
print('binary_label: ', binary_label)

patient_FRS_dict = matdata11['patient_Structure_Regs_dict']  # (12,12)
networks = patient_FRS_dict['connection_map'][0][0]-np.eye(134)
node_position = patient_FRS_dict['roi_Centroid'][0][0]
node_feature = patient_FRS_dict['node_features'][0][0]
node_id = patient_FRS_dict['label_list'][0][0][0]

t1 = time.time()
print(t1-t)

# ignore_nodes = None
ignore_nodes = (2,3,30)
if ignore_nodes is not None:
    networks = np.delete(networks, ignore_nodes, axis=0)
    networks = np.delete(networks, ignore_nodes, axis=1)
    node_feature = np.delete(node_feature, ignore_nodes, axis=0)
    node_position = np.delete(node_position, ignore_nodes, axis=0)
    node_id = node_id[:-len(ignore_nodes)]

show_graph_with_labels(networks, node_position, node_id)
print('node_feature: ', np.shape(node_feature), node_feature)
print(np.argmax(np.sum(infarct_ROI, axis=(1, 2))))

fig1, ax1 = plt.subplots(nrows=2, ncols=3)
ax1[0,0].imshow(np.squeeze(DTI_MD[47]), cmap='gray',)
ax1[0,1].imshow(np.squeeze(DTI_Trace[47]), cmap='gray')
ax1[0,2].imshow(np.squeeze(Flair[80]), cmap='gray')
ax1[1,0].imshow(np.squeeze(infarct_ROI[47]))
ax1[1,1].imshow(np.squeeze(WMH_ROI[80]))
ax1[1,2].axis('off')

fig2, ax2 = plt.subplots(nrows=2, ncols=3)
ax2[0,0].imshow(np.squeeze(DTI_MD_d2[40]), cmap='gray',)
ax2[0,1].imshow(np.squeeze(DTI_Trace_d2[40]), cmap='gray')
ax2[0,2].imshow(np.squeeze(Flair_d2[40]), cmap='gray')
ax2[1,0].imshow(np.squeeze(infarct_ROI_d2[40]))
ax2[1,1].imshow(np.squeeze(WMH_ROI_d2[40]))
ax2[1,2].axis('off')

rgb_vol = np.concatenate((DTI_MD[:,..., np.newaxis], DTI_Trace[:,..., np.newaxis], Flair[:,..., np.newaxis]), axis=-1).astype(np.int8)
rgb_vol_d2 = np.concatenate((DTI_MD_d2[:,..., np.newaxis], DTI_Trace_d2[:,..., np.newaxis], Flair_d2[:,..., np.newaxis]), axis=-1).astype(np.int8)

fig3, ax3 = plt.subplots(nrows=2, ncols=3)
ax3[0,0].imshow(np.squeeze(rgb_vol[46]))
ax3[0,1].imshow(np.squeeze(rgb_vol[80]))
ax3[0,2].imshow(np.squeeze(rgb_vol[120]))
ax3[1,0].imshow(np.squeeze(rgb_vol_d2[24]))
ax3[1,1].imshow(np.squeeze(rgb_vol_d2[40]))
ax3[1,2].imshow(np.squeeze(rgb_vol_d2[60]))
plt.show()
