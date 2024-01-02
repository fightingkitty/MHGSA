import SimpleITK as sitk
from decimal import Decimal
import random
import os
import numpy as np
import yaml
from yaml.loader import SafeLoader
from radiomics import firstorder, glcm, shape
import six
from pathlib import Path


def extract_radiomics_feature(image, mask, paras_url, label, graylevel=False, _shape=True, _firstorder=True, _glcm=True):
    if graylevel:
        orig_img = sitk.GetArrayFromImage(image)
        sitk_image_orig = sitk.GetImageFromArray(orig_img)
        image_graylevel = (orig_img - np.min(orig_img[:]))
        sitk_image_graylevel = sitk.GetImageFromArray(image_graylevel * 255.0)

        sitk_mask = sitk.GetImageFromArray(sitk.GetArrayFromImage(mask))
    else:
        sitk_image_graylevel = image
        sitk_image_orig = image
        sitk_mask = sitk.GetImageFromArray(sitk.GetArrayFromImage(mask))

    with open(paras_url) as f:
        data = yaml.load(f, Loader=SafeLoader)
    settings = data['settings']
    featureClass = data['featureclass']
    settings['label'] = label
    radiomics_feature = dict()
    if _shape:
        shapeFeatures = shape.RadiomicsShape(sitk_image_orig, sitk_mask, **settings)
        shapeFeatures.disableAllFeatures()
        for name in featureClass['shape']:
            shapeFeatures.enableFeatureByName(str(name))
        results = shapeFeatures.execute()
        for (key, val) in six.iteritems(results):
            radiomics_feature['shape_' + str(key)] = float(val)
    if _firstorder:
        firstorderFeatures = firstorder.RadiomicsFirstOrder(sitk_image_orig, sitk_mask, **settings)
        firstorderFeatures.disableAllFeatures()
        for name in featureClass['firstorder']:
            firstorderFeatures.enableFeatureByName(str(name))
        results = firstorderFeatures.execute()
        for (key, val) in six.iteritems(results):
            radiomics_feature['firstorder_' + str(key)] = float(val)

    if _glcm:
        glcmFeatures = glcm.RadiomicsGLCM(sitk_image_graylevel, sitk_mask, **settings)
        glcmFeatures.disableAllFeatures()
        for name in featureClass['glcm']:
            glcmFeatures.enableFeatureByName(str(name))
        results = glcmFeatures.execute()
        for (key, val) in six.iteritems(results):
            radiomics_feature['glcm_' + str(key)] = float(val)

    return radiomics_feature


def Structure_Regs_poolfunc(label, sitk_Structure_Regs, sitk_DTI_MD, sitk_DTI_Trace, sitk_Flair, paras_structure_tissue, patient_intracranial_volume):
    dilate_filter = sitk.BinaryDilateImageFilter()
    dilate_filter.SetKernelRadius(1)
    dilate_filter.SetKernelType(sitk.sitkBall)
    dilate_filter.SetForegroundValue(1)

    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(sitk_Structure_Regs)
    sitk_roi = (sitk_Structure_Regs == label)

    Is_too_small = CheckMask(sitk_roi, label=1)
    if Is_too_small:
        roi_Centroid = None
        npy_connected_roi_list = None
        node_features = None
        return label, roi_Centroid, npy_connected_roi_list, node_features

    sitk_dilated_roi = dilate_filter.Execute(sitk_roi)
    npy_dilated_connected_roi = sitk.GetArrayFromImage(sitk_dilated_roi-sitk_roi)*sitk.GetArrayFromImage(sitk_Structure_Regs)
    npy_connected_roi_list = np.unique(npy_dilated_connected_roi[:])
    npy_connected_roi_list = npy_connected_roi_list[npy_connected_roi_list > 0]-1
    center_gravity_coordiate = sitk_roi.TransformPhysicalPointToIndex(label_shape_filter.GetCentroid(label))
    roi_Centroid = np.asarray(center_gravity_coordiate)
    FR_volume = np.sum(sitk.GetArrayFromImage(sitk_roi)[:])/patient_intracranial_volume

    node_shape_feature = extract_radiomics_feature(sitk_DTI_MD, sitk_Structure_Regs, paras_structure_tissue,
                                                   label=label, graylevel=True, _firstorder=False, _glcm=False)
    node_MD_features = extract_radiomics_feature(sitk_DTI_MD, sitk_Structure_Regs, paras_structure_tissue,
                                                 label=label, graylevel=True, _shape=False)
    node_Trace_features = extract_radiomics_feature(sitk_DTI_Trace, sitk_Structure_Regs, paras_structure_tissue,
                                                    label=label, graylevel=True, _shape=False)
    node_Flair_features = extract_radiomics_feature(sitk_Flair, sitk_Structure_Regs, paras_structure_tissue,
                                                    label=label, graylevel=True, _shape=False)
    node_features = [FR_volume]
    for val in node_shape_feature.values():
        node_features.append(val)
    for val in node_MD_features.values():
        node_features.append(val)
    for val in node_Trace_features.values():
        node_features.append(val)
    for val in node_Flair_features.values():
        node_features.append(val)
    return label, roi_Centroid, npy_connected_roi_list, node_features


def CheckMask(mask, label):
    roi_mask = (mask == label)
    if np.sum(sitk.GetArrayFromImage(roi_mask)[:]) < 4:
        return True
    return False


def resample_volume(sitk_volume, new_spacing, interpolator):
    original_spacing = sitk_volume.GetSpacing()
    original_size = sitk_volume.GetSize()
    new_size = [int(Decimal(str(osz*ospc/nspc)).quantize(Decimal("1"), rounding="ROUND_HALF_UP"))
                for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]
    return sitk.Resample(sitk_volume, new_size, sitk.Transform(), interpolator,
                         sitk_volume.GetOrigin(), new_spacing, sitk_volume.GetDirection(), 0,
                         sitk_volume.GetPixelID())


def load_dataset_from_txt(url):
    mat_files = []
    for p in url if isinstance(url, list) else [url]:
        p = Path(p)
        if p.is_file():
            with open(p, 'r') as t:
                t = t.read().strip().splitlines()
                mat_files += [x for x in t]
    print("{0:d} mat files are found from {1}!".format(len(mat_files), url))
    return mat_files


def split(all_list, shuffle=False, ratio=0.8):
    num = len(all_list)
    offset = int(num * ratio)
    if num == 0 or offset < 1:
        return [], all_list
    if shuffle:
        random.shuffle(all_list)  # random the file list
    train = all_list[:offset]
    test = all_list[offset:]
    return train, test


def five_folder_split(all_list, folder_id=1):
    num = len(all_list)
    folder_size = int(num // 5)
    folder = dict()
    folder[1] = all_list[0:folder_size]
    folder[2] = all_list[folder_size:folder_size*2]
    folder[3] = all_list[folder_size*2:folder_size*3]
    folder[4] = all_list[folder_size*3:folder_size*4]
    folder[5] = all_list[folder_size*4:]

    train = []
    for i in range(1, 6):
        if i == folder_id:
            continue
        else:
            train += folder[i]
    test = folder[folder_id]
    return train, test


def isexist_file(path_list):
    for path in path_list:
        if not os.path.exists(path):
            print('Can not find ', path)
            return False
    return True