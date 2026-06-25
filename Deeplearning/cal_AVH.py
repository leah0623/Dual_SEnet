import os
import numpy as np
import SimpleITK as sitk
from os.path import join
import pandas as pd

def get_calnum(cam_arr, select_bool):
    cambin = np.arange(0, 1, 0.01)
    roi_cam = cam_arr[select_bool]
    cum_dvh = np.zeros([1, len(cambin)])

    if len(roi_cam) != 0:
        for num in range(len(cambin)):
            cum_dvh[0, num] = len(np.where(roi_cam > cambin[num])[0]) / len(roi_cam) * 100
    return cum_dvh

base_dir = os.path.join('./experiments', 'Dual_task_CD', 'Grad_CAM')
patient_list = os.listdir(base_dir)

cambin = np.arange(0, 1, 0.05)
dosebin = np.arange(40, 75, 5)

for event in ['DM', 'LR']:
    cam_dvhall = []
    save_files = os.path.join('./experiments', 'Dual_task_CD', 'gradcam_cal_{}.xlsx'.format(event))

    for patient in patient_list:
        cam_one = []

        grad_cam = sitk.ReadImage(join(base_dir, patient, 'CAM_layer3{}.nii.gz'.format(event)))
        dose = sitk.ReadImage(join(base_dir, patient, 'Dose.nii.gz'))
        mask = sitk.ReadImage(join(base_dir, patient, 'Mask.nii.gz'), sitk.sitkInt16)
        erode_filter = sitk.BinaryDilateImageFilter()
        erode_filter.SetKernelRadius(1)
        dilatemask = erode_filter.Execute(mask)

        cam_arr = sitk.GetArrayFromImage(grad_cam)
        dose_arr = sitk.GetArrayFromImage(dose) * 87
        mask_arr = sitk.GetArrayFromImage(mask)
        dilatemask_arr = sitk.GetArrayFromImage(dilatemask)

        g0index = mask_arr == 1
        g3index = dilatemask_arr == 1
        g01index = (~g0index) & g3index

        gindex_list = [
            g0index, g3index, g01index
        ]

        for gindex in gindex_list:
            for dosen in dosebin:
                dosenumflag = dose_arr >= dosen
                findex = dosenumflag & gindex

        cam_one = np.concatenate(cam_one, axis=1)
        cam_dvhall.append(cam_one)
    camdvhGD = np.concatenate(cam_dvhall, axis=0)
    colnames = []
    for gname in ['0', '3', '03']:
        for dosen in dosebin:
            colnames.extend(list(map(lambda name: 'G{}_D{}_W{:.2f}'.format(gname, dosen, name), cambin)))
    camdvhGD_pd = pd.DataFrame(camdvhGD, index=patient_list, columns=colnames)
    camdvhGD_pd.to_excel(save_files)