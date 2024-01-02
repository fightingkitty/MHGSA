import pandas as pd
import copy
import os
import SimpleITK as sitk
import numpy as np
from multiprocessing import Pool
from functools import partial
import time
from utils import CheckMask, Structure_Regs_poolfunc, extract_radiomics_feature


class Basedataset(object):

    def __init__(self, filepath, sheet_name=None):

        self.filepath = filepath
        self.sheet_name = sheet_name
        self.data = self.read_excel()
        self.interest_column = ['ID',
                                'Alter',
                                'sex',
                                'education_years_r',
                                'bmsmok_akt_rr',
                                'bmalk_a_rr',
                                'bmhyperton_r',
                                'bmdm_r',
                                'bmvhf_r',
                                'bmbmi',
                                'bmLDL_Cholesterin_mg_dl',
                                'bmstr_r',
                                'bmnihss_total',
                                'bmmrs_vor',
                                'bmmoca_mmse_r',
                                'bmdiagkat',
                                'bmi_toast_r',
                                'bmiqcode_total',
                                'FU12_cog_average_full',
                                'FU12_group_neu_full']
        self.features_normalized = ['Alter_n',
                       'sex_n',
                       'education_years_r_n',
                       'bmsmok_akt_rr',
                       'bmalk_a_rr',
                       'bmhyperton_r',
                       'bmdm_r',
                       'bmvhf_r',
                       'bmbmi',
                       'bmLDL_Cholesterin_mg_dl',
                       'bmstr_r',
                       'bmnihss_total',
                       'bmmrs_vor',
                       'bmmoca_mmse_r',
                       'bmdiagkat_A',
                       'bmdiagkat_B',
                       'bmi_toast_r',
                       'bmiqcode_total']

    def read_excel(self):
        return pd.read_excel(self.filepath, sheet_name=self.sheet_name)

    def get_interested_colums(self, tmp):
        data = copy.copy(tmp)
        if self.interest_column is not None:
            return data[self.interest_column]
        else:
            return data

    def normalization_DEMDAS_DEDEMAS_dataset(self, tmp):
        data = copy.copy(tmp)
        data = data.dropna(subset=['FU12_cog_average_full', 'FU12_group_neu_full'])
        print('there might be ', data.shape[0], 'samples can be used in this dataset')
        data['bmdiagkat'] = data['bmdiagkat'].replace(3, 2)
        data['bmdiagkat_A'] = -data['bmdiagkat'] + 2
        data['bmdiagkat_B'] = data['bmdiagkat'] - 1
        data['sex_n'] = data['sex'] * 2 - 1.0

        data.loc[:, 'bmmrs_vor'] = data['bmmrs_vor'] / 5.0
        data['bmiqcode_total'] = (data['bmiqcode_total'] - 16.0) / 64.0
        data['bmi_toast_r'] = data['bmi_toast_r'] / 5.0
        data['education_years_r_n'] = data['education_years_r'] / 12.0 - 1.0
        data['Alter_n'] = data['Alter'] / 100.0
        data['bmbmi'] = data['bmbmi'] / 25.0 - 1.0
        data['bmLDL_Cholesterin_mg_dl'] = data['bmLDL_Cholesterin_mg_dl'] / 130.0 - 1.0
        data['bmnihss_total'] = data['bmnihss_total'] / 42.0

        return data

    def get_ID(self, data):
        return data['ID'].to_list()

    def get_binary_label(self, data):
        return data['FU12_group_neu_full'].to_numpy()

    def get_regression_label(self, data):
        return data['FU12_cog_average_full'].to_numpy()

    def get_missing_vector(self, data):
        return data[self.features_normalized].isnull().to_numpy().astype('int')

    def get_tabular_data(self, data):
        return data[self.features_normalized].fillna(0.0).to_numpy()


class StrokeDataset(Basedataset):
    def __init__(self, filepath, sheet_name=None):
        super().__init__(filepath, sheet_name)
        self.analysis_data = self.get_interested_colums(self.data)
        self.analysis_data_normalized = self.normalization_DEMDAS_DEDEMAS_dataset(self.analysis_data)

        self.ID = self.get_ID(self.analysis_data_normalized)
        self.binarylabel = self.get_binary_label(self.analysis_data_normalized)
        self.regressionlabel = self.get_regression_label(self.analysis_data_normalized)
        self.miss_vector = self.get_missing_vector(self.analysis_data_normalized)
        self.tabular_data = self.get_tabular_data(self.analysis_data_normalized)


class Sample(object):
    def __init__(self, id, root):
        patient_foldername = str(id)
        self.patient_folder_url = os.path.join(root, patient_foldername)
        commen_url = os.path.join(self.patient_folder_url, 'M0-MNI-RIGID')

        self.patient_foldername = patient_foldername
        self.commen_url = commen_url
        self.DTI_MD_path = os.path.join(commen_url, patient_foldername + '-M0-DTI_EC_DIFF_MD.nii.gz')
        self.DTI_Trace_path = os.path.join(commen_url, patient_foldername + '-M0-DTI_EC_trace_unbiased.nii.gz')
        self.FLAIR_path = os.path.join(commen_url, patient_foldername + '-M0-FLAIR_N4.nii.gz')

        self.infarct_mask_path = os.path.join(commen_url, patient_foldername + '-M0-infarct_c2.nii.gz')
        self.WMH_mask_path = os.path.join(commen_url, patient_foldername + '-M0-WMH_c2.nii.gz')

        self.Brain_mask_path = os.path.join(commen_url, patient_foldername + '-M0-T1_MALPEM_tissues_m2.nii.gz')
        self.T1_mask_path = os.path.join(commen_url, patient_foldername + '-M0-T1_mask_m2.nii.gz')
        self.T1_Structure_Regs_path = os.path.join(commen_url, patient_foldername + '-M0-T1_MALPEM_m2.nii.gz')  # brain structure region segmentation

        self.paras_infart_wmh = '../Data/paras_infarct_MWH.yaml'
        self.paras_structure_tissue = '../Data/paras_structure_tissue.yaml'
        self.exist = self.isexist_file()
        if self.exist:
            self.patient_intracranial_volume = None
            self.patient_tissue1_volume = None
            self.patient_tissue2_volume = None
            self.patient_tissue3_volume = None
            self.patient_tissue4_volume = None
            self.patient_tissue5_volume = None
            self.patient_tissues_volume = None

            self.patient_infarct_volume = None
            self.patient_WMH_volume = None

            self.infarct_vol_ratio = 0
            self.infarct_nums = 0
            self.infarct_roi_var = 0
            self.infarct_radiomics_feature = None

            self.WMH_vol_ratio = 0
            self.WMH_nums = 0
            self.WMH_roi_var = 0
            self.WMH_radiomics_feature = None

            self.brain_vol_ratio = 0

            self.patient_Structure_Regs_dict = dict()
            self.patient_global_features = None

            self.npy_tissues_volume = None
            self.npy_infarct_volume = None
            self.npy_wmh_volume = None

            self.sitk_infarct_volume = None
            self.sitk_Structure_Regs = None
            self.sitk_wmh_volume = None
            self.sitk_DTI_MD = None
            self.sitk_DTI_Trace = None
            self.sitk_Flair = None

    def isexist_file(self):
        path_list = [self.patient_folder_url,
                     self.DTI_MD_path,
                     self.DTI_Trace_path,
                     self.FLAIR_path,
                     self.infarct_mask_path,
                     self.WMH_mask_path,
                     self.Brain_mask_path,
                     self.WMH_mask_path,
                     self.T1_mask_path,
                     self.T1_Structure_Regs_path]
        for path in path_list:
            if not os.path.exists(path):
                # print('Can not find ', path)
                return False
        return True

    def Read_intracranial_volume(self):
        #  patient intracranial volume
        sitk_intracranial_volume = sitk.ReadImage(self.T1_mask_path, sitk.sitkInt32)
        npy_intracranial_volume = sitk.GetArrayFromImage(sitk_intracranial_volume)
        npy_intracranial_volume[npy_intracranial_volume > 0] = 1.0
        self.patient_intracranial_volume = np.sum(npy_intracranial_volume[:])

    def Read_braintissues_volume(self):
        # brain tissues volume
        sitk_tissues_volume = sitk.ReadImage(self.Brain_mask_path, sitk.sitkInt32)
        npy_tissues_volume = sitk.GetArrayFromImage(sitk_tissues_volume)
        npy_tissue1_volume = np.zeros_like(npy_tissues_volume)
        npy_tissue2_volume = np.zeros_like(npy_tissues_volume)
        npy_tissue3_volume = np.zeros_like(npy_tissues_volume)
        npy_tissue4_volume = np.zeros_like(npy_tissues_volume)
        npy_tissue5_volume = np.zeros_like(npy_tissues_volume)

        npy_tissue1_volume[npy_tissues_volume == 1.0] = 1
        npy_tissue2_volume[npy_tissues_volume == 2.0] = 1
        npy_tissue3_volume[npy_tissues_volume == 3.0] = 1
        npy_tissue4_volume[npy_tissues_volume == 4.0] = 1
        npy_tissue5_volume[npy_tissues_volume == 5.0] = 1
        npy_tissues_volume[npy_tissues_volume > 0] = 1

        self.patient_tissue1_volume = np.sum(npy_tissue1_volume[:])
        self.patient_tissue2_volume = np.sum(npy_tissue2_volume[:])
        self.patient_tissue3_volume = np.sum(npy_tissue3_volume[:])
        self.patient_tissue4_volume = np.sum(npy_tissue4_volume[:])
        self.patient_tissue5_volume = np.sum(npy_tissue5_volume[:])
        self.patient_tissues_volume = np.sum(npy_tissues_volume[:])

        self.npy_tissues_volume = npy_tissues_volume   # real brain mask without CSF

    def Read_infarct_volume(self):
        #  infarct volume
        self.sitk_infarct_volume = sitk.ReadImage(self.infarct_mask_path, sitk.sitkInt32)
        npy_infarct_volume = sitk.GetArrayFromImage(self.sitk_infarct_volume)
        npy_infarct_volume[npy_infarct_volume > 0] = 1.0
        npy_infarct_volume = npy_infarct_volume * self.npy_tissues_volume
        self.npy_infarct_volume = npy_infarct_volume
        self.patient_infarct_volume = np.sum(npy_infarct_volume[:])

    def Read_WHM_volume(self):
        self.sitk_wmh_volume = sitk.ReadImage(self.WMH_mask_path, sitk.sitkInt32)
        npy_wmh_volume = sitk.GetArrayFromImage(self.sitk_wmh_volume)
        npy_wmh_volume[npy_wmh_volume > 0] = 1.0
        npy_wmh_volume = npy_wmh_volume * self.npy_tissues_volume
        self.npy_wmh_volume = npy_wmh_volume
        self.patient_WMH_volume = np.sum(npy_wmh_volume[:])

    def Read_volumes(self):
        self.sitk_DTI_MD = sitk.ReadImage(self.DTI_MD_path, sitk.sitkFloat32)
        self.sitk_DTI_Trace = sitk.ReadImage(self.DTI_Trace_path, sitk.sitkFloat32)
        self.sitk_Flair = sitk.ReadImage(self.FLAIR_path, sitk.sitkFloat32)
        self.sitk_Structure_Regs = sitk.ReadImage(self.T1_Structure_Regs_path, sitk.sitkInt32)
        self.res_Volumes()

    def res_Volumes(self):
        npy_Structure_Regs = sitk.GetArrayFromImage(self.sitk_Structure_Regs)
        npy_Structure_Regs[npy_Structure_Regs == 14] = 0
        npy_Structure_Regs[npy_Structure_Regs == 15] = 0
        npy_Structure_Regs[npy_Structure_Regs == 33] = 0
        npy_Structure_Regs[npy_Structure_Regs == 34] = 0
        npy_Structure_Regs[(npy_Structure_Regs >= 16) & (npy_Structure_Regs <= 32)] = \
            npy_Structure_Regs[(npy_Structure_Regs >= 16) & (npy_Structure_Regs <= 32)] - 2
        npy_Structure_Regs[npy_Structure_Regs >= 35] = npy_Structure_Regs[npy_Structure_Regs >= 35] - 4
        res_Structure_Regs = sitk.GetImageFromArray(npy_Structure_Regs)
        res_Structure_Regs.SetSpacing(self.sitk_Structure_Regs.GetSpacing())
        self.sitk_Structure_Regs = res_Structure_Regs

        npy_DTI_MD = (sitk.GetArrayFromImage(self.sitk_DTI_MD).astype(np.float32)-0.00035)/0.000631
        res_DTI_M = sitk.GetImageFromArray(npy_DTI_MD)
        res_DTI_M.SetSpacing(self.sitk_DTI_MD.GetSpacing())
        self.sitk_DTI_MD = res_DTI_M

        npy_DTI_Trace = (sitk.GetArrayFromImage(self.sitk_DTI_Trace).astype(np.float32)-39.197)/66.4376
        res_DTI_Trace = sitk.GetImageFromArray(npy_DTI_Trace)
        res_DTI_Trace.SetSpacing(self.sitk_DTI_Trace.GetSpacing())
        self.sitk_DTI_Trace = res_DTI_Trace

        npy_Flair = (sitk.GetArrayFromImage(self.sitk_Flair).astype(np.float32)-41.257)/67.5937
        res_Flair = sitk.GetImageFromArray(npy_Flair)
        res_Flair.SetSpacing(self.sitk_Flair.GetSpacing())
        self.sitk_Flair = res_Flair

    def infarct_analysis(self):
        Is_too_small = CheckMask(self.sitk_infarct_volume, label=1)
        if Is_too_small:
            self.infarct_vol_ratio = 0
            self.infarct_nums = 0
            self.infarct_roi_var = 0
            self.infarct_radiomics_feature = np.zeros(18)
            return 0
        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        cca = sitk.ConnectedComponentImageFilter()
        cca.SetFullyConnected(True)

        self.infarct_vol_ratio = self.patient_infarct_volume / self.patient_intracranial_volume
        #  infarct roi counts and distribution var
        sitk_infarct_rois = cca.Execute(self.sitk_infarct_volume)
        self.infarct_nums = cca.GetObjectCount()
        label_shape_filter.Execute(sitk_infarct_rois)
        infarct_roi_position = []
        for label in range(1, label_shape_filter.GetNumberOfLabels() + 1):
            center_gravity_coordiate = label_shape_filter.GetCentroid(label)
            infarct_roi_position.append(np.asarray(center_gravity_coordiate))
        infarct_roi_position = np.asarray(infarct_roi_position)
        infarct_roi_var = 0
        if infarct_roi_position.shape[0] > 1:
            infarct_roi_var = np.sum(np.sqrt(np.sum((infarct_roi_position - np.mean(infarct_roi_position, axis=0)) ** 2,
                                                    axis=1)), axis=0) / infarct_roi_position.shape[0]
        self.infarct_roi_var = infarct_roi_var

        dict_infarct_radiomics_feature = extract_radiomics_feature(self.sitk_DTI_MD, self.sitk_infarct_volume,
                                                                   self.paras_infart_wmh, label=1, graylevel=True)

        self.infarct_radiomics_feature = [self.infarct_vol_ratio, self.infarct_nums/20.0, self.infarct_roi_var/100.0]
        for val in dict_infarct_radiomics_feature.values():
            self.infarct_radiomics_feature.append(val)

    def WHM_analysis(self):

        Is_too_small = CheckMask(self.sitk_wmh_volume, label=1)
        if Is_too_small:
            self.WMH_vol_ratio = 0
            self.WMH_nums = 0
            self.WMH_roi_var = 0
            self.WMH_radiomics_feature = np.zeros(18)
            return 0
        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        cca = sitk.ConnectedComponentImageFilter()
        cca.SetFullyConnected(True)

        self.WMH_vol_ratio = self.patient_WMH_volume / self.patient_intracranial_volume
        #  WMH roi counts and distribution var
        sitk_WMH_rois = cca.Execute(self.sitk_wmh_volume)
        self.WMH_nums = cca.GetObjectCount()
        label_shape_filter.Execute(sitk_WMH_rois)
        WMH_roi_position = []
        for label in range(1, label_shape_filter.GetNumberOfLabels() + 1):
            center_gravity_coordiate = label_shape_filter.GetCentroid(label)
            WMH_roi_position.append(np.asarray(center_gravity_coordiate))
        WMH_roi_position = np.asarray(WMH_roi_position)
        WMH_roi_var = 0
        if WMH_roi_position.shape[0] > 1:
            WMH_roi_var = np.sum(np.sqrt(np.sum((WMH_roi_position - np.mean(WMH_roi_position, axis=0)) ** 2,
                                                axis=1)), axis=0) / WMH_roi_position.shape[0]
        self.WMH_roi_var = WMH_roi_var
        dict_WMH_radiomics_feature = extract_radiomics_feature(self.sitk_DTI_MD, self.sitk_wmh_volume,
                                                               self.paras_infart_wmh, label=1, graylevel=True)

        self.WMH_radiomics_feature = [self.WMH_vol_ratio, self.WMH_nums/20.0, self.WMH_roi_var/100.0]
        for val in dict_WMH_radiomics_feature.values():
            self.WMH_radiomics_feature.append(val)

    def Structure_Regs_analysis(self):
        npy_Structure_Regs = sitk.GetArrayFromImage(self.sitk_Structure_Regs)

        patient_Structure_Regs_connection_map = np.eye(134)
        label_list = np.unique(npy_Structure_Regs[:])
        label_list = label_list[label_list > 0]
        self.patient_Structure_Regs_dict['label_list'] = label_list
        roi_Centroid = np.zeros([134, 3])
        node_intensity_feature = np.zeros([134, 45])
        pool = Pool(16)
        # result = pool.map(self.Structure_Regs_poolfunc, label_list.tolist())
        prod_x = partial(Structure_Regs_poolfunc, sitk_Structure_Regs=self.sitk_Structure_Regs, sitk_DTI_MD=self.sitk_DTI_MD,
                         sitk_DTI_Trace=self.sitk_DTI_Trace, sitk_Flair=self.sitk_Flair,
                         paras_structure_tissue=self.paras_structure_tissue,
                         patient_intracranial_volume=self.patient_intracranial_volume)
        result = pool.map(prod_x, label_list.tolist())
        pool.close()

        for index, v in enumerate(label_list.tolist()):
            info = result[index]
            if info[1] is not None:
                roi_Centroid[v - 1, :] = info[1]
            if info[2] is not None:
                patient_Structure_Regs_connection_map[v - 1, info[2]] = 1.0
            if info[3] is not None:
                node_intensity_feature[v-1, :] = info[3]

        self.patient_Structure_Regs_dict['roi_Centroid'] = roi_Centroid
        self.patient_Structure_Regs_dict['connection_map'] = patient_Structure_Regs_connection_map

        # histogram analysis
        npy_infarct_FR = self.npy_infarct_volume * npy_Structure_Regs
        npy_WMH_FR = self.npy_wmh_volume * npy_Structure_Regs

        Structure_Regs_bincount = np.bincount(npy_Structure_Regs.reshape([-1]))[1:]
        infarct_FR_bincount = np.bincount(npy_infarct_FR.reshape([-1]))[1:]
        WMH_FR_bincount = np.bincount(npy_WMH_FR.reshape([-1]))[1:]

        Structure_Regs_matrix_count = np.zeros([134], dtype=np.float32)
        Structure_Regs_matrix_count[:Structure_Regs_bincount.shape[0]] = Structure_Regs_bincount
        Structure_Regs_matrix_count[Structure_Regs_matrix_count == 0] = 1.0

        infarct_matrix_barcode = np.zeros([134], dtype=np.float32)
        infarct_matrix_barcode[:infarct_FR_bincount.shape[0]] = infarct_FR_bincount
        patient_infarct_barcode = infarct_matrix_barcode / Structure_Regs_matrix_count  # input1,  don't need normalization

        WMH_matrix_barcode = np.zeros([134], dtype=np.float32)
        WMH_matrix_barcode[:WMH_FR_bincount.shape[0]] = WMH_FR_bincount
        patient_WMH_barcode = WMH_matrix_barcode / Structure_Regs_matrix_count

        patient_FR_node_feature = np.concatenate(
            (patient_infarct_barcode[:, np.newaxis], patient_WMH_barcode[:, np.newaxis], node_intensity_feature), axis=1)
        self.patient_Structure_Regs_dict['node_features'] = patient_FR_node_feature

    def Process(self):
        self.Read_intracranial_volume()
        self.Read_braintissues_volume()
        self.Read_WHM_volume()
        self.Read_infarct_volume()
        self.Read_volumes()

        self.infarct_analysis()
        self.WHM_analysis()
        self.brain_vol_ratio = self.patient_tissues_volume / self.patient_intracranial_volume
        self.Structure_Regs_analysis()
        self.patient_global_features = np.append([self.brain_vol_ratio],
                                                 np.append(self.WMH_radiomics_feature, self.infarct_radiomics_feature))


if __name__ == '__main__':
    root = '/u/home/lius/Dataset/stroke/DEMDAS/DEMDAS-preproc2'
    excelfile = '../Data/Tabular_data.xlsx'
    DEDEMAS = StrokeDataset(excelfile, sheet_name='DEMDAS')
    # print(DEDEMAS.binarylabel)
    # print(DEDEMAS.regressionlabel)
    # print(DEDEMAS.miss_vector)
    # print(DEDEMAS.tabular_data)
    ids = DEDEMAS.ID
    t0 = time.time()
    patient_data = Sample('P_DD_BBF_2IA8Z4EU', root)
    if patient_data.exist:
        patient_data.Process()
    t1 = time.time()
    print(t1-t0)
    print(patient_data.patient_global_features)
    print(patient_data.WMH_radiomics_feature)
    print(patient_data.infarct_radiomics_feature)
    print(patient_data.patient_Structure_Regs_dict['node_features'][0])








