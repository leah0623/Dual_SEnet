import numpy as np
import SimpleITK as sitk
import torch
import os
import torchio as tio
import pandas as pd

def read_follow_data(filepath, sheetname):
    ExcelFile = pd.read_excel(filepath, sheet_name=sheetname, header=None)
    name = np.array(ExcelFile[0])
    event_DM = np.array(ExcelFile[1]).astype(np.float32)
    time_DM = np.array(ExcelFile[2]).astype(np.float32)
    event_LR = np.array(ExcelFile[4]).astype(np.float32)
    time_LR = np.array(ExcelFile[5]).astype(np.float32)
    return name, event_DM, time_DM, event_LR, time_LR


def read_data(data_dir, patient, event_DM, time_DM, event_LR, time_LR):
    subjects_list = []
    for i in range(0, len(patient)):
        ct_nii = sitk.ReadImage(os.path.join(data_dir, patient[i], 'CT.nii.gz'), sitk.sitkFloat32)
        ct_arr = sitk.GetArrayFromImage(ct_nii)
        ct_arr = np.expand_dims(ct_arr, axis=0)
        ct_arr = np.transpose(ct_arr, axes=(0, 2, 3, 1))
        ct_tensor = torch.from_numpy(ct_arr)
        dose_nii = sitk.ReadImage(os.path.join(data_dir, patient[i], 'DOSE.nii.gz'), sitk.sitkFloat32)
        dose_arr = sitk.GetArrayFromImage(dose_nii)
        dose_arr = np.expand_dims(dose_arr, axis=0)
        dose_arr = np.transpose(dose_arr, axes=(0, 2, 3, 1))
        dose_tensor = torch.from_numpy(dose_arr)
        mask_nii = sitk.ReadImage(os.path.join(data_dir, patient[i], 'GTV_mask.nii.gz'), sitk.sitkFloat32)
        mask_arr = sitk.GetArrayFromImage(mask_nii)
        mask_arr = np.expand_dims(mask_arr, axis=0)
        mask_arr = np.transpose(mask_arr, axes=(0, 2, 3, 1))
        mask_tensor = torch.from_numpy(mask_arr)

        # torchio.Subject 一种用于存储与Subject相关联的图像以及处理所需的任何其他元数据的数据结构。
        subjects_list.append(
            tio.Subject(ct=tio.ScalarImage(tensor=ct_tensor), dose=tio.ScalarImage(tensor=dose_tensor),
                        mask=tio.LabelMap(tensor=mask_tensor),
                        target_DM=event_DM[i], target_LR=event_LR[i],
                        event_time_DM=time_DM[i], event_time_LR=time_LR[i],
                        name=patient[i]))

    return subjects_list

def get_weight(event_w):
    weight0 = len(event_w) / len(np.where(event_w == 0)[0])
    weight1 = len(event_w) / len(np.where(event_w == 1)[0])
    weight2 = min(len(event_w) / len(np.where(event_w == 2)[0]), 8)

    weight = []
    for i in range(len(event_w)):
        if event_w[i] == 0:
            weight.append(weight0)
        elif event_w[i] == 1:
            weight.append(weight1)
        elif event_w[i] == 2:
            weight.append(weight2)

    return np.array(weight)

def get_transform(mode='train'):
    if mode=='train':
        return tio.Compose([
            tio.Clamp(out_min=-400, out_max=400),
            tio.CropOrPad(target_shape=(128, 128, 64), padding_mode='edge', mask_name='mask'),
            tio.RandomFlip(axes=0, flip_probability=0.5),
            tio.RandomFlip(axes=1, flip_probability=0.5),
            tio.RandomAffine(scales=0, degrees=(0, 0, 30), translation=30, isotropic=True, p=0.8),
            tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=(-400, 400), include=['ct']),
            tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=(0, 87), include=['dose']),
        ])
    else:
        return tio.Compose([
            tio.Clamp(out_min=-400, out_max=400),
            tio.CropOrPad(target_shape=(128, 128, 64), padding_mode='edge', mask_name='mask'),
            tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=(-400, 400), include=['ct']),
            tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=(0, 87), include=['dose']),
        ])
