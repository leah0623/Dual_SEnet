import os
from os.path import join
import SimpleITK as sitk
import BaseTools as btools

base_data_dir = r'D:\project\0_Competition\data\Original_data'
patient_list = os.listdir(base_data_dir)
save_dir = r'D:\project\0_Competition\data\new_data\CT_dose_resample_333'
new_spacing = [3.0, 3.0, 3.0]

for name in patient_list:
    print('-------------------- {} --------------------'.format(name))

    os.makedirs(join(save_dir, name), exist_ok=True)

    # load CT
    CT_path = join(base_data_dir, name, 'CT')
    series = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(CT_path)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(series)
    image = reader.Execute()

    print('CT spacing:', image.GetSpacing())
    print('CT size:', image.GetSize())
    print('CT origin:', image.GetOrigin())

    # load GTV mask
    mask_list = btools.RS2nii(join(save_dir, name, 'CT.nii.gz'), join(base_data_dir, name, 'RS'), mask_key=['gtv'])
    mask_arr = None
    for mask in mask_list:
        if mask_arr is None:
            mask_arr = sitk.GetArrayFromImage(mask)
        else:
            mask_arr += sitk.GetArrayFromImage(mask)
    mask_arr[mask_arr != 0] = 1
    GTV_mask = sitk.GetImageFromArray(mask_arr)
    GTV_mask.CopyInformation(mask_list[0])

    # load dose
    RD_dcm_path = btools.get_match_dose(base_data_dir, name)
    if len(RD_dcm_path) != 1:
        dose_arr = None
        for file in RD_dcm_path:
            dose = sitk.ReadImage(join(base_data_dir, name, 'RD', file))
            dose = btools.scaling_dose(dose, join(base_data_dir, name, 'RD', file))
            print('dose {} load'.format(file))
            print('dose spacing:', dose.GetSpacing())
            print('dose size:', dose.GetSize())
            print('dose origin:', dose.GetOrigin())

            dose = btools.check_shape_and_resize(dose, image)

            if dose_arr is None:
                dose_arr = sitk.GetArrayFromImage(dose)
            else:
                dose_arr += sitk.GetArrayFromImage(dose)
        new_dose = sitk.GetImageFromArray(dose_arr)
        new_dose.CopyInformation(dose)
    else:
        new_dose = sitk.ReadImage(join(base_data_dir, name, 'RD', RD_dcm_path[0]))
        new_dose = btools.scaling_dose(new_dose, join(base_data_dir, name, 'RD', RD_dcm_path[0]))
        print('dose {} load'.format(RD_dcm_path[0]))
        print('dose spacing:', new_dose.GetSpacing())
        print('dose size:', new_dose.GetSize())
        print('dose origin:', new_dose.GetOrigin())
        new_dose = btools.check_shape_and_resize(new_dose, image)

    # resample
    image_new = btools.resample_image(image, out_spacing=new_spacing, resamplemethod=sitk.sitkLinear)
    dose_new = btools.resample_image(dose, out_spacing=new_spacing, resamplemethod=sitk.sitkLinear)
    mask_new = btools.resample_image(GTV_mask, out_spacing=new_spacing, resamplemethod=sitk.sitkNearestNeighbor)

    sitk.WriteImage(image_new, join(save_dir, name, 'CT.nii.gz'))
    sitk.WriteImage(dose_new, join(save_dir, name, 'DOSE.nii.gz'))
    sitk.WriteImage(mask_new, join(save_dir, name, 'GTV_mask.nii.gz'))