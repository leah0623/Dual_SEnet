import os
import SimpleITK as sitk
from os.path import join

base_dir = r'D:\project\0_Competition\data\new_data\CT_dose_resample_333'

for patient in os.listdir(base_dir):
    mask = sitk.ReadImage(join(base_dir, patient, 'GTV_mask.nii.gz'), sitk.sitkInt16)
    # explan 3mm
    erode_filter = sitk.BinaryDilateImageFilter()
    erode_filter.SetKernelRadius(1)
    dilatemask = erode_filter.Execute(mask)
    sitk.WriteImage(dilatemask, join(base_dir, patient, 'Mask_ex3mm.nii.gz'))

    mask_arr = sitk.GetArrayFromImage(mask)
    dilatemask_arr = sitk.GetArrayFromImage(dilatemask)
    ring_arr = dilatemask_arr - mask_arr
    ringmask = sitk.GetImageFromArray(ring_arr)
    ringmask.CopyInformation(mask)
    sitk.WriteImage(ringmask, join(base_dir, patient, 'Mask_ring3mm.nii.gz'))