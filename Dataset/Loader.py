import torch
from pathlib import Path
from torch.utils.data import Dataset
import torchio as tio
import scipy.io as sio
import random
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class Stroke_Dataset(Dataset):
    def __init__(self, datasetpath_list, augment=True, is_shuffle=False, ignore_nodes=None, norm=True, type='tabular'):
        self.type = type
        self.data = self.load_dataset_from_txt(datasetpath_list)

        if isinstance(datasetpath_list, list):
            self.norm_mat_path = Path(datasetpath_list[0]).parent / 'norm_mfs_mat.mat'
        else:
            self.norm_mat_path = Path(datasetpath_list).parent / 'norm_mfs_mat.mat'

        print(self.norm_mat_path)
        matdata = sio.loadmat(str(self.norm_mat_path))
        self.mean_tabular = np.squeeze(matdata['Mean_tabular'])
        self.std_tabular = np.squeeze(matdata['Std_tabular'])
        self.mean_node = np.squeeze(matdata['Mean_node'])
        self.std_node = np.squeeze(matdata['Std_node'])
        self.norm = norm

        if is_shuffle:
            random.shuffle(self.data)
        self.is_aug = augment

        transforms_dict = {
            tio.RandomNoise(): 0.5,
            tio.RandomAffine(): 0.5,
        }

        self.Img_spatial_transforms = tio.OneOf(transforms_dict, p=0.5)
        self.ignore_nodes = ignore_nodes
        self.selected_tabularfeature_dicts={
            'Alter_n': 0,  # should norm /100.0
            'sex_n': 1,
            'education_years_r_n': 2,  # should norm /12.0 - 1.0
            'bmsmok_akt_rr': 3,
            'bmalk_a_rr': 4,
            'bmhyperton_r': 5,
            'bmdm_r': 6,
            'bmvhf_r': 7,
            'bmbmi': 8,
            'bmLDL_Cholesterin_mg_dl': 9,
            'bmstr_r': 10,
            'bmnihss_total': 11,
            'bmmrs_vor': 12,
            'bmmoca_mmse_r': 13,
            # 'bmdiagkat_A': 14,
            # 'bmdiagkat_B': 15,
            # 'bmi_toast_r': 16,
            # 'bmiqcode_total': 17,
            # 'brain_vol_ratio': 18,
            # 'WMH_vol_ratio': 19,
            # 'WMH_nums': 20,
            # 'WMH_roi_var': 21,
            # 'WMH_SurfaceVolumeRatio': 22,  # shape
            # 'WMH_MD_10Percentile': 23,  # first order
            # 'WMH_MD_90Percentile': 24,
            # 'WMH_MD_Mean': 25,
            # 'WMH_MD_Median': 26,
            # 'WMH_MD_Minimum': 27,
            # 'WMH_MD_Maximum': 28,
            # 'WMH_MD_Variance': 29,
            # 'WMH_MD_Skewness': 30,
            # 'WMH_MD_Kurtosis': 31,
            # 'WMH_MD_InterquartileRange': 32,
            # 'WMH_MD_Correlation': 33,  # glcm
            # 'WMH_MD_DifferenceEntropy': 34,
            # 'WMH_MD_JointEnergy': 35,
            # 'WMH_MD_Imc2': 36,
            'infarct_vol_ratio': 37,
            # 'infarct_nums': 38,
            # 'infarct_roi_var': 39,
            # 'infarct_SurfaceVolumeRatio': 40,  # shape
            # 'infarct_MD_10Percentile': 41,  # first order
            # 'infarct_MD_90Percentile': 42,
            # 'infarct_MD_Mean': 43,
            # 'infarct_MD_Median': 44,
            # 'infarct_MD_Minimum': 45,
            # 'infarct_MD_Maximum': 46,
            # 'infarct_MD_Variance': 47,
            # 'infarct_MD_Skewness': 48,
            # 'infarct_MD_Kurtosis': 49,
            # 'infarct_MD_InterquartileRange': 50,
            # 'infarct_MD_Correlation': 51,  # glcm
            # 'infarct_MD_DifferenceEntropy': 52,
            # 'infarct_MD_JointEnergy': 53,
            # 'infarct_MD_Imc2': 54,
            # 'SVDscore_total': 55,
            'lacune_count': 56,
            'PVS_level': 57,
            'Fazekas_PVWM': 58,
            'Fazekas_DWM': 59,
            'CMB_count': 60

        }
        self.selected_nodefeature_dicts={
            'infarct_ratio_in_node': 0,
            'WMH_ratio_in_node': 1,
            'node_vol_ratio': 2,

            # 'SurfaceVolumeRatio': 3,
            # 'LeastAxisLength': 4,
            # 'MD_10Percentile': 5,
            # 'MD_90Percentile': 6,

            'MD_Mean': 7,
            'MD_Median': 8,
            'MD_Minimum': 9,
            'MD_Maximum': 10,
            'MD_Variance': 11,

            # 'MD_Skewness': 12,
            # 'MD_Kurtosis': 13,  # 异常
            # 'MD_InterquartileRange': 14,
            # 'MD_Correlation': 15,
            # 'MD_DifferenceEntropy': 16,
            # 'MD_JointEnergy': 17,
            # 'MD_Imc2': 18,
            # 'Trace_10Percentile': 19,
            # 'Trace_90Percentile': 20,

            'Trace_Mean': 21,
            'Trace_Median': 22,
            'Trace_Minimum': 23,
            'Trace_Maximum': 24,
            'Trace_Variance': 25,

            # 'Trace_Skewness': 26,
            # 'Trace_Kurtosis': 27,  # 异常
            # 'Trace_InterquartileRange': 28,
            # 'Trace_Correlation': 29,
            # 'Trace_DifferenceEntropy': 30,
            # 'Trace_JointEnergy': 31,
            # 'Trace_Imc2': 32,
            # 'Flair_10Percentile': 33,
            # 'Flair_90Percentile': 34,

            'Flair_Mean': 35,
            'Flair_Median': 36,
            'Flair_Minimum': 37,
            'Flair_Maximum': 38,
            'Flair_Variance': 39,

            # 'Flair_Skewness': 40,
            # 'Flair_Kurtosis': 41,  # 异常
            # 'Flair_InterquartileRange': 42,
            # 'Flair_Correlation': 43,
            # 'Flair_DifferenceEntropy': 44,
            # 'Flair_JointEnergy': 45,
            # 'Flair_Imc2': 46
        }

        # self.Img_Transform_MD = transforms.Normalize(mean=0.00035, std=0.000631)
        # self.Img_Transform_Trace = transforms.Normalize(mean=39.197, std=66.4376)
        # self.Img_Transform_Flair = transforms.Normalize(mean=41.257, std=67.5937)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):

        patient_id = self.data[item].split('/')[-1]

        if self.type == 'tabular':
            tupledata = self.load_tabular_data(item, self.is_aug)
            label = self.load_label(item)

            binary_label = self.AsTensor(label)
            clinical_Tabular = self.AsTensor(tupledata[1])
            missing_vector = self.AsTensor(tupledata[0])

            adj_matrix = self.AsTensor(tupledata[2])
            nood_features = self.AsTensor(tupledata[3])

            sample = {'binary_label': binary_label,
                      'clinical_Tabular': clinical_Tabular, 'missing_vector': missing_vector,
                      'adj_matrix': adj_matrix, 'nood_features': nood_features,
                      'patient_id': patient_id}
            return sample

        elif self.type == 'volume':
            tupledata = self.load_volume_data(item)
            label = self.load_label(item)

            binary_label = self.AsTensor(label)
            DTI_MD = self.AsTensor(tupledata[0])
            DTI_Trace = self.AsTensor(tupledata[1])
            Flair = self.AsTensor(tupledata[2])
            infarct_ROI = self.AsTensor(tupledata[3])
            WMH_ROI = self.AsTensor(tupledata[4])
            if self.is_aug:
                subject_dict = tio.Subject(
                    DTI_MD=tio.ScalarImage(tensor=DTI_MD),
                    DTI_Trace=tio.ScalarImage(tensor=DTI_Trace),
                    Flair=tio.ScalarImage(tensor=Flair),
                    infarct_ROI=tio.LabelMap(tensor=infarct_ROI),
                    WMH_ROI=tio.LabelMap(tensor=WMH_ROI),
                )
                transformed_dic = self.Img_spatial_transforms(subject_dict)
                DTI_MD = transformed_dic['DTI_MD'][tio.DATA]
                DTI_Trace = transformed_dic['DTI_Trace'][tio.DATA]
                Flair = transformed_dic['Flair'][tio.DATA]
                infarct_ROI = transformed_dic['infarct_ROI'][tio.DATA]
                WMH_ROI = transformed_dic['WMH_ROI'][tio.DATA]

            # DTI_MD = self.Img_Transform_MD(DTI_MD)
            # DTI_Trace = self.Img_Transform_Trace(DTI_Trace)
            # Flair = self.Img_Transform_Flair(Flair)

            sample = {'binary_label': binary_label, 'infarct_mask': infarct_ROI, 'WMH_mask': WMH_ROI,
                      'DTI_MD': DTI_MD, 'DTI_Trace': DTI_Trace, 'Flair': Flair,
                      'patient_id': patient_id}

            return sample

        elif self.type == 'all':
            tupledata1 = self.load_tabular_data(item,self.is_aug)
            label = self.load_label(item)

            binary_label = self.AsTensor(label)
            clinical_Tabular = self.AsTensor(tupledata1[1])
            missing_vector = self.AsTensor(tupledata1[0])

            adj_matrix = self.AsTensor(tupledata1[2])
            nood_features = self.AsTensor(tupledata1[3])

            tupledata2 = self.load_volume_data(item)

            DTI_MD = self.AsTensor(tupledata2[0])
            DTI_Trace = self.AsTensor(tupledata2[1])
            Flair = self.AsTensor(tupledata2[2])
            infarct_ROI = self.AsTensor(tupledata2[3])
            WMH_ROI = self.AsTensor(tupledata2[4])
            if self.is_aug:
                subject_dict = tio.Subject(
                    DTI_MD=tio.ScalarImage(tensor=DTI_MD),
                    DTI_Trace=tio.ScalarImage(tensor=DTI_Trace),
                    Flair=tio.ScalarImage(tensor=Flair),
                    infarct_ROI=tio.LabelMap(tensor=infarct_ROI),
                    WMH_ROI=tio.LabelMap(tensor=WMH_ROI),
                )
                transformed_dic = self.Img_spatial_transforms(subject_dict)
                DTI_MD = transformed_dic['DTI_MD'][tio.DATA]
                DTI_Trace = transformed_dic['DTI_Trace'][tio.DATA]
                Flair = transformed_dic['Flair'][tio.DATA]

                infarct_ROI = transformed_dic['infarct_ROI'][tio.DATA]
                WMH_ROI = transformed_dic['WMH_ROI'][tio.DATA]

            sample = {'binary_label': binary_label,
                      'clinical_Tabular': clinical_Tabular, 'missing_vector': missing_vector,
                      'adj_matrix': adj_matrix, 'nood_features': nood_features,
                      'infarct_mask': infarct_ROI, 'WMH_mask': WMH_ROI,
                      'DTI_MD': DTI_MD, 'DTI_Trace': DTI_Trace, 'Flair': Flair,
                      'patient_id': patient_id}
            return sample
        else:
            print("ERROR: please give a right type, the value from {'tabular','volume','all'} ")
            exit(0)

    def load_dataset_from_txt(self, url):
        mat_files = []
        for p in url if isinstance(url, list) else [url]:
            p = Path(p)
            if p.is_file():
                with open(p, 'r') as t:
                    t = t.read().strip().splitlines()
                    mat_files += [x for x in t]
        print("{0:d} mat files are found from {1}!".format(len(mat_files), url))
        return mat_files

    def load_tabular_data(self, index, aug=False):

        folder_url = self.data[index]
        mat_path = Path(folder_url)/'Tabular_data_manualfeatures.mat'
        matdata = sio.loadmat(mat_path)

        Tabular_data_missing = np.squeeze(matdata['Tabular_data_missing'], axis=0).astype(np.float32)  # (105)
        Tabular_data = np.squeeze(matdata['Tabular_data'], axis=0).astype(np.float32)  # (105)
        patient_Structure_Regs_dict = matdata['patient_Structure_Regs_dict']
        adjacency_matrix = (patient_Structure_Regs_dict['connection_map'][0][0]).astype(
            np.float32)  # (134, 134)
        node_feature = patient_Structure_Regs_dict['node_features'][0][0].astype(np.float32)  # (134,129)

        if aug:
            node_feature = self.add_noise_Node(node_feature)
            Tabular_data = self.add_noise_Tabular(Tabular_data)

        if self.norm:
            Tabular_data = (Tabular_data-self.mean_tabular)/self.std_tabular
            node_feature = (node_feature-self.mean_node)/self.std_node

        if self.ignore_nodes is not None:
            adjacency_matrix = np.delete(adjacency_matrix, self.ignore_nodes, axis=0)
            adjacency_matrix = np.delete(adjacency_matrix, self.ignore_nodes, axis=1)
            node_feature = np.delete(node_feature,self.ignore_nodes,axis=0)

        selected_node_feature_index = list(self.selected_nodefeature_dicts.values())
        if len(selected_node_feature_index):
            node_feature = node_feature[:, selected_node_feature_index]
        selected_tabular_feature_index = list(self.selected_tabularfeature_dicts.values())
        if len(selected_tabular_feature_index):
            Tabular_data = Tabular_data[selected_tabular_feature_index]
            Tabular_data_missing = Tabular_data_missing[selected_tabular_feature_index]

        return Tabular_data_missing, Tabular_data, adjacency_matrix, node_feature

    def load_volume_data(self, index, read_downsample=False):
        folder_url = self.data[index]
        add_suffix = ''
        if read_downsample:
            add_suffix = '_d2'
        mat_path_MD = Path(folder_url)/('DTI_MD' + add_suffix + '.mat')
        mat_path_Trace = Path(folder_url)/('DTI_Trace' + add_suffix + '.mat')
        mat_path_Flair = Path(folder_url)/('Flair' + add_suffix + '.mat')
        mat_path_infarct_ROI = Path(folder_url)/('infarct_ROI' + add_suffix + '.mat')
        mat_path_WMH_ROI = Path(folder_url)/('WMH_ROI' + add_suffix + '.mat')

        matdata_MD = sio.loadmat(mat_path_MD)
        matdata_Trace = sio.loadmat(mat_path_Trace)
        matdata_Flair = sio.loadmat(mat_path_Flair)
        matdata_infarct_ROI = sio.loadmat(mat_path_infarct_ROI)
        matdata_WMH_ROI = sio.loadmat(mat_path_WMH_ROI)

        DTI_MD = self.AddChannelDim(matdata_MD['DTI_MD'].astype(np.float32))  # (1, 159,190,153) [C D H W]
        DTI_Trace = self.AddChannelDim(matdata_Trace['DTI_Trace'].astype(np.float32))  # (1, 159,190,153)
        Flair = self.AddChannelDim(matdata_Flair['Flair'].astype(np.float32))  # (1, 159,190,153)

        infarct_ROI = self.AddChannelDim(matdata_infarct_ROI['infarct_ROI'].astype(np.float32))   # (159,190,153)
        WMH_ROI = self.AddChannelDim(matdata_WMH_ROI['WMH_ROI'].astype(np.float32))   # (159,190,153)
        return DTI_MD, DTI_Trace, Flair, infarct_ROI, WMH_ROI

    def load_label(self, index):
        folder_url = self.data[index]
        mat_path = Path(folder_url) / 'label.mat'
        matdata = sio.loadmat(mat_path)
        binary_label = np.squeeze(matdata['binary_label'], axis=0).astype(np.float32)  # (1)
        return binary_label

    def add_noise_Tabular(self, Tabular_data, p=0.5):
        if np.random.random() < p:
            noise = np.random.randn(Tabular_data.shape[0]) * self.std_tabular * 0.1
            Tabular_data = noise + Tabular_data
            return Tabular_data
        else:
            return Tabular_data

    def add_noise_Node(self, nodefeatures, p=0.5):
        if np.random.random() < p:
            noise = np.random.randn(nodefeatures.shape[0],nodefeatures.shape[1]) * self.std_node * 0.1
            nodefeatures = noise + nodefeatures
            return nodefeatures
        else:
            return nodefeatures

    def AsTensor(self, npy):
        return torch.from_numpy(npy)


    def AddChannelDim(self, img):
        return img[np.newaxis]


if __name__ == '__main__':
    data_path = '/u/home/lius/Project/Stroke_Radiomics/Dataset/Data/strokenormdataset/DEMDAS_train_folder1.txt'
    traindataloader = DataLoader(Stroke_Dataset(data_path, augment=True, is_shuffle=True, ignore_nodes=(2,3,30), norm=True, type='tabular'),
                                 batch_size=1, shuffle=True, num_workers=8, drop_last=True)
    for i, batch in enumerate(traindataloader):
        print(i)
        print(batch['binary_label'].shape)
        print(batch['clinical_Tabular'].shape)
        print(batch['missing_vector'].shape)

        print(batch['clinical_Tabular'])

        print(batch['adj_matrix'].shape)
        print(batch['nood_features'].shape)
        print(batch['nood_features'][0,0])
        exit(0)

        # if torch.isnan(batch['nood_features']).any():
        #     print('as')


        # print(batch['DTI_MD'].shape)
        # print(batch['DTI_Trace'].shape)
        # print(batch['Flair'].shape)



    # for i in range(len(stroke_dataset)):
    #     sample = stroke_dataset[i]
    #     print(sample['Volume_data'].shape)
    #
    #
    #     Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    #     input_Volume = Tensor(1, 3, 159, 190, 153)
    #     input_ROI1 = Tensor(1, 1, 159, 190, 153)
    #     input_ROI2 = Tensor(1, 1, 159, 190, 153)
    #
    #
    #     input_Volume = input_Volume.copy_(sample['Volume_data'])
    #     input_ROI1 = input_ROI1.copy_(sample['infarct_mask'])
    #     input_ROI2 = input_ROI2.copy_(sample['WMH_mask'])
    #
    #     img1 = np.squeeze(input_Volume.to('cpu').numpy())
    #     img2 = np.squeeze(input_ROI1.to('cpu').numpy())
    #     img3 = np.squeeze(input_ROI2.to('cpu').numpy())
    #
    #     fig, ax = plt.subplots(nrows=2, ncols=3)
    #     ax[0, 0].imshow(np.squeeze(img1[0, 40]), cmap='gray')
    #     ax[0, 1].imshow(np.squeeze(img1[1, 40]), cmap='gray')
    #     ax[0, 2].imshow(np.squeeze(img1[2, 40]), cmap='gray')
    #     ax[1, 0].imshow(np.squeeze(img2[40]))
    #     ax[1, 1].imshow(np.squeeze(img3[40]))
    #     ax[1, 2].axis('off')
    #     plt.show()