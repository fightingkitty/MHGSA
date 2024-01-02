from utils import five_folder_split
import random
import glob
from pathlib import Path
import scipy.io as sio
import numpy as np
import os

sheet_name = 'DEMDAS'
dataset_root = '/home/shuting/Datasets/'+sheet_name+'_preproc_liu_norm'

file_list = glob.glob(str(Path(dataset_root)/'*/label.mat'))
mat_folder_0 = []
mat_folder_1 = []
for file in file_list:
    matdata = sio.loadmat(file)
    binary_label = np.squeeze(matdata['binary_label'], axis=0)
    if binary_label == 0:
        mat_folder_0.append(str(Path(file).parent))
    elif binary_label == 1:
        mat_folder_1.append(str(Path(file).parent))
    else:
        print("error label, check folder", Path(file).parent.name)

if not os.path.exists('../Data/strokenormdataset'):
    os.makedirs('../Data/strokenormdataset')

txt_all_1 = open('../Data/strokenormdataset/' + sheet_name + '_all_1.txt', 'w')
txt_all_0 = open('../Data/strokenormdataset/' + sheet_name + '_all_0.txt', 'w')
for file in mat_folder_1:
    txt_all_1.write(file + '\n')
txt_all_1.close()

for file in mat_folder_0:
    txt_all_0.write(file + '\n')
txt_all_0.close()

random.shuffle(mat_folder_0)  # random the file list
random.shuffle(mat_folder_1)  # random the file list

traindataset_l0_f1, testdataset_l0_f1 = five_folder_split(mat_folder_0, 1)
traindataset_l0_f2, testdataset_l0_f2 = five_folder_split(mat_folder_0, 2)
traindataset_l0_f3, testdataset_l0_f3 = five_folder_split(mat_folder_0, 3)
traindataset_l0_f4, testdataset_l0_f4 = five_folder_split(mat_folder_0, 4)
traindataset_l0_f5, testdataset_l0_f5 = five_folder_split(mat_folder_0, 5)

traindataset_l1_f1, testdataset_l1_f1 = five_folder_split(mat_folder_1, 1)
traindataset_l1_f2, testdataset_l1_f2 = five_folder_split(mat_folder_1, 2)
traindataset_l1_f3, testdataset_l1_f3 = five_folder_split(mat_folder_1, 3)
traindataset_l1_f4, testdataset_l1_f4 = five_folder_split(mat_folder_1, 4)
traindataset_l1_f5, testdataset_l1_f5 = five_folder_split(mat_folder_1, 5)

traindataset_folder = dict()
testdataset_folder = dict()

traindataset_folder[1] = traindataset_l0_f1 + traindataset_l1_f1
testdataset_folder[1] = testdataset_l0_f1 + testdataset_l1_f1

traindataset_folder[2] = traindataset_l0_f2 + traindataset_l1_f2
testdataset_folder[2] = testdataset_l0_f2 + testdataset_l1_f2

traindataset_folder[3] = traindataset_l0_f3 + traindataset_l1_f3
testdataset_folder[3] = testdataset_l0_f3 + testdataset_l1_f3

traindataset_folder[4] = traindataset_l0_f4 + traindataset_l1_f4
testdataset_folder[4] = testdataset_l0_f4 + testdataset_l1_f4

traindataset_folder[5] = traindataset_l0_f5 + traindataset_l1_f5
testdataset_folder[5] = testdataset_l0_f5 + testdataset_l1_f5

for i in range(5):
    txt_test = open('../Data/strokenormdataset/' + sheet_name + '_test_folder'+str(i+1)+'.txt', 'w')
    txt_train = open('../Data/strokenormdataset/' + sheet_name + '_train_folder'+str(i+1)+'.txt', 'w')

    random.shuffle(traindataset_folder[i+1])  # random the file list
    random.shuffle(testdataset_folder[i+1])  # random the file list

    for file in traindataset_folder[i+1]:
        txt_train.write(file + '\n')
    txt_train.close()

    for file in testdataset_folder[i+1]:
        txt_test.write(file + '\n')
    txt_test.close()



