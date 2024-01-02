import SimpleITK as sitk
import os
import numpy as np
import scipy.io as sio
from utils import resample_volume
from tqdm import tqdm
from Datasets import StrokeDataset, Sample
import shutil
from pathlib import Path


def process_sample(index, img_id, root, Tabular, Tabular_missing,
                   binary_label, regression_label):

    patient_data = Sample(img_id, root)
    if patient_data.exist:
        # get tablular data and missing vector
        patient_clinical_Tabular_data = Tabular[index, :]  # input_0
        patient_clinical_Tabular_data_missing = Tabular_missing[index, :]
        # two labels
        patient_binary_label = binary_label[index]  # GT1
        patient_regression_label = regression_label[index]  # GT2

        patient_data.Process()

        patient_clinical_Tabular_data = np.append(patient_clinical_Tabular_data,
                                                  patient_data.patient_global_features)
        patient_clinical_Tabular_data_missing = np.append(patient_clinical_Tabular_data_missing,
                                                          np.zeros(len(patient_data.patient_global_features)))

        patient_DTI_MD = sitk.GetArrayFromImage(patient_data.sitk_DTI_MD)  # (80, 95, 77) (C,D,H,W)
        patient_DTI_MD_resampled = sitk.GetArrayFromImage(
            resample_volume(patient_data.sitk_DTI_MD, new_spacing=[2.0, 2.0, 2.0], interpolator=sitk.sitkLinear))

        patient_DTI_Trace = sitk.GetArrayFromImage(patient_data.sitk_DTI_Trace)  # (80, 95, 77) (C,D,H,W)
        patient_DTI_Trace_resampled = sitk.GetArrayFromImage(
            resample_volume(patient_data.sitk_DTI_Trace, new_spacing=[2.0, 2.0, 2.0], interpolator=sitk.sitkLinear))

        patient_Flair = sitk.GetArrayFromImage(patient_data.sitk_Flair)  # (80, 95, 77) (C,D,H,W)
        patient_Flair_resampled = sitk.GetArrayFromImage(
            resample_volume(patient_data.sitk_Flair, new_spacing=[2.0, 2.0, 2.0], interpolator=sitk.sitkLinear))

        patient_infarct_ROI = sitk.GetArrayFromImage(patient_data.sitk_infarct_volume)
        patient_infarct_ROI_resampled = sitk.GetArrayFromImage(
            resample_volume(patient_data.sitk_infarct_volume, new_spacing=[2.0, 2.0, 2.0],
                            interpolator=sitk.sitkNearestNeighbor))
        patient_infarct_ROI_resampled[patient_infarct_ROI_resampled > 0] = 1.0

        patient_WMH_ROI = sitk.GetArrayFromImage(patient_data.sitk_wmh_volume)
        patient_WMH_ROI_resampled = sitk.GetArrayFromImage(
            resample_volume(patient_data.sitk_wmh_volume, new_spacing=[2.0, 2.0, 2.0],
                            interpolator=sitk.sitkNearestNeighbor))
        patient_WMH_ROI_resampled[patient_WMH_ROI_resampled > 0] = 1.0
        # volumes
        mat_DTIMD_dict = {"DTI_MD": patient_DTI_MD.astype(np.float32)}
        mat_Trace_dict = {"DTI_Trace": patient_DTI_Trace.astype(np.float32)}
        mat_Flair_dict = {"Flair": patient_Flair.astype(np.float32)}
        mat_infarct_ROI_dict = {"infarct_ROI": patient_infarct_ROI.astype(np.int32)}
        mat_WMH_ROI_dict = {"WMH_ROI": patient_WMH_ROI.astype(np.int32)}
        # volumes after downsample
        mat_DTIMD_dict_d2 = {"DTI_MD": patient_DTI_MD_resampled.astype(np.float32)}
        mat_Trace_dict_d2 = {"DTI_Trace": patient_DTI_Trace_resampled.astype(np.float32)}
        mat_Flair_dict_d2 = {"Flair": patient_Flair_resampled.astype(np.float32)}
        mat_infarct_ROI_dict_d2 = {"infarct_ROI": patient_infarct_ROI_resampled.astype(np.int32)}
        mat_WMH_ROI_dict_d2 = {"WMH_ROI": patient_WMH_ROI_resampled.astype(np.int32)}

        # Tabular data
        mat_tabuler_dict = {"Tabular_data": patient_clinical_Tabular_data.astype(np.float32),
                            "Tabular_data_missing": patient_clinical_Tabular_data_missing.astype(np.int32),
                            "patient_Structure_Regs_dict": patient_data.patient_Structure_Regs_dict}

        mat_label_dict = {"binary_label": patient_binary_label.astype(np.float32),
                          "regression_label": patient_regression_label.astype(np.float32)}

        save_floder_url = str(Path(root).parent / (sheet_name + '_preproc_liu_norm') / patient_data.patient_foldername)

        if os.path.exists(save_floder_url):
            shutil.rmtree(save_floder_url)
            os.makedirs(save_floder_url)
        else:
            os.makedirs(save_floder_url)

        mat_filepath_MD = os.path.join(save_floder_url, 'DTI_MD.mat')
        mat_filepath_Trace = os.path.join(save_floder_url, 'DTI_Trace.mat')
        mat_filepath_Flair = os.path.join(save_floder_url, 'Flair.mat')
        mat_filepath_infarctROI = os.path.join(save_floder_url, 'infarct_ROI.mat')
        mat_filepath_WMHROI = os.path.join(save_floder_url, 'WMH_ROI.mat')

        mat_filepath_MD_d2 = os.path.join(save_floder_url, 'DTI_MD_d2.mat')
        mat_filepath_Trace_d2 = os.path.join(save_floder_url, 'DTI_Trace_d2.mat')
        mat_filepath_Flair_d2 = os.path.join(save_floder_url, 'Flair_d2.mat')
        mat_filepath_infarctROI_d2 = os.path.join(save_floder_url, 'infarct_ROI_d2.mat')
        mat_filepath_WMHROI_d2 = os.path.join(save_floder_url, 'WMH_ROI_d2.mat')

        mat_filepath_Tabular = os.path.join(save_floder_url, 'Tabular_data.mat')
        mat_filepath_label = os.path.join(save_floder_url, 'label.mat')

        sio.savemat(mat_filepath_MD, mat_DTIMD_dict)
        sio.savemat(mat_filepath_Trace, mat_Trace_dict)
        sio.savemat(mat_filepath_Flair, mat_Flair_dict)
        sio.savemat(mat_filepath_infarctROI, mat_infarct_ROI_dict)
        sio.savemat(mat_filepath_WMHROI, mat_WMH_ROI_dict)

        sio.savemat(mat_filepath_MD_d2, mat_DTIMD_dict_d2)
        sio.savemat(mat_filepath_Trace_d2, mat_Trace_dict_d2)
        sio.savemat(mat_filepath_Flair_d2, mat_Flair_dict_d2)
        sio.savemat(mat_filepath_infarctROI_d2, mat_infarct_ROI_dict_d2)
        sio.savemat(mat_filepath_WMHROI_d2, mat_WMH_ROI_dict_d2)

        sio.savemat(mat_filepath_Tabular, mat_tabuler_dict)
        sio.savemat(mat_filepath_label, mat_label_dict)


def creat_dataset(root, excelfile, sheet_name, resumeid=0):
    #  load dataset
    Table = StrokeDataset(excelfile, sheet_name=sheet_name)

    ID = Table.ID
    binary_label = Table.binarylabel
    regression_label = Table.regressionlabel
    # a 0-1 index for value-missing
    Tabular_missing = Table.miss_vector
    Tabular = Table.tabular_data

    log_file_url = '../Data/' + sheet_name + 'creat_log.txt'
    if resumeid == 0:
        if os.path.exists(log_file_url):
            os.remove(log_file_url)
    ID = ID[resumeid:]
    for index, img_id in tqdm(enumerate(ID), total=ID.__len__()):
        fid = index+resumeid
        try:
            process_sample(fid, img_id, root, Tabular, Tabular_missing,
                           binary_label, regression_label)
            with open(log_file_url, 'a') as f:
                record_line = "ID: %d, Foldername: %s --> processed well" % (fid, img_id)
                f.write(record_line + '\n')
                f.write('\n')

        except Exception:
            with open(log_file_url, 'a') as f:
                record_line = "ID: %d, Foldername: %s --> not processed well" % (fid, img_id)
                f.write(record_line + '\n')
                f.write('\n')


def Creat_single_sample(root, excelfile, sheet_name,samplename):
    #  load dataset
    Table = StrokeDataset(excelfile, sheet_name=sheet_name)

    ID = Table.ID
    binary_label = Table.binarylabel
    regression_label = Table.regressionlabel
    # a 0-1 index for value-missing
    Tabular_missing = Table.miss_vector
    Tabular = Table.tabular_data
    index = ID.index(samplename)
    check_sample_url = str(Path(root).parent / (sheet_name + '_preproc_liu') / samplename)
    process_sample(index, samplename, root, Tabular, Tabular_missing, binary_label, regression_label)

    if os.path.exists(check_sample_url):
        print('this sample already exits')
    else:
        print('do it')
        process_sample(index, samplename, root, Tabular, Tabular_missing, binary_label, regression_label)
        print('finished')


if __name__ == '__main__':
    # root = '/u/home/lius/Dataset/stroke/DEMDAS/DEMDAS-preproc2'
    # excelfile = '../Data/Tabular_data.xlsx'
    # sheet_name = 'DEMDAS'
    # Creat_single_sample(root, excelfile, sheet_name, samplename='P_DD_BBF_2IA8Z4EU')

    root = '/u/home/lius/Dataset/stroke/DEMDAS/DEMDAS-preproc2'
    excelfile = '../Data/Tabular_data.xlsx'
    sheet_name = 'DEMDAS'
    creat_dataset(root, excelfile, sheet_name)

    # root = '/u/home/lius/Dataset/stroke/DEDEMAS/DEDEMAS-preproc2'
    # excelfile = '../Data/Tabular_data.xlsx'
    # sheet_name = 'DEDEMAS'
    # creat_dataset(root, excelfile, sheet_name)

