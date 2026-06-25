import os
from os.path import join
import numpy as np
import SimpleITK as sitk
import pydicom
import glob
from scipy import ndimage
from skimage import draw
import logging

def ContourOutOfBoundsException():
    pass

def get_match_dose(base_data_dir, name):
    struct_data = glob.glob(join(base_data_dir, name, 'RS', '*.dcm'))
    str_dcm = pydicom.read_file(struct_data[0])
    dose_data = glob.glob(join(base_data_dir, name, 'RD', '*.dcm'))
    plan_data = glob.glob(join(base_data_dir, name, 'RP', '*.dcm'))
    if len(plan_data) != 0:
        march_plan = []
        for plan_file in plan_data:
            plan_dcm = pydicom.read_file(plan_file)
            if plan_dcm.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID == str_dcm.SOPInstanceUID:
                march_plan.append(plan_dcm.SOPInstanceUID)

    dose_list = []
    if len(dose_data) != 1:
        for dose_file in dose_data:
            dose_dcm = pydicom.read_file(dose_file)
            if hasattr(dose_dcm, 'ReferencedStructureSetSequence'):
                if dose_dcm.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID == str_dcm.SOPInstanceUID:
                    dose_list.append(dose_file.split('\\')[-1])
            elif hasattr(dose_dcm, 'ReferencedRTPlanSequence'):
                if dose_dcm.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID in march_plan:
                    dose_list.append(dose_file.split('\\')[-1])
    else:
        dose_list.append(dose_data[0].split('\\')[-1])

    return dose_list

def scaling_dose(dose, dose_path):
    # 获取剂量的缩放因子
    dose_py = pydicom.read_file(dose_path, force=True)
    doseScaling = dose_py.DoseGridScaling

    dose_arr = sitk.GetArrayFromImage(dose)
    s_dose_arr = dose_arr * doseScaling
    s_dose = sitk.GetImageFromArray(s_dose_arr)

    s_dose.CopyInformation(dose)

    return s_dose

def resize_image_itk(ori_img, target_img, resamplemethod=sitk.sitkNearestNeighbor):
    """
    用itk方法将原始图像resample到与目标图像一致
    :param ori_img: 原始需要对齐的itk图像
    :param target_img: 要对齐的目标itk图像
    :param resamplemethod: itk插值方法: sitk.sitkLinear-线性  sitk.sitkNearestNeighbor-最近邻
    :return:img_res_itk: 重采样好的itk图像
    """
    target_Size = target_img.GetSize()      # 目标图像大小  [x,y,z]
    target_Spacing = target_img.GetSpacing()   # 目标的体素块尺寸    [x,y,z]
    target_origin = target_img.GetOrigin()      # 目标的起点 [x,y,z]
    target_direction = target_img.GetDirection()  # 目标的方向 [冠,矢,横]=[z,y,x]

    # itk的方法进行resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ori_img)  # 需要重新采样的目标图像
    # 设置目标图像的信息
    resampler.SetSize(target_Size)		# 目标图像大小
    resampler.SetOutputOrigin(target_origin)
    resampler.SetOutputDirection(target_direction)
    resampler.SetOutputSpacing(target_Spacing)
    # 根据需要重采样图像的情况设置不同的dype
    if resamplemethod == sitk.sitkNearestNeighbor:
        resampler.SetOutputPixelType(sitk.sitkUInt16)   # 近邻插值用于mask的，保存uint16
    else:
        resampler.SetOutputPixelType(sitk.sitkFloat32)  # 线性插值用于PET/CT/MRI之类的，保存float32
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itk_img_resampled = resampler.Execute(ori_img)  # 得到重新采样后的图像
    return itk_img_resampled

def resample_image(itk_image, out_spacing=[3.0, 3.0, 3.0], resamplemethod=sitk.sitkLinear):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    # 根据输出out_spacing设置新的size
    out_size = [
        int(np.round(original_size[0] * original_spacing[0] / out_spacing[0])),
        int(np.round(original_size[1] * original_spacing[1] / out_spacing[1])),
        int(np.round(original_size[2] * original_spacing[2] / out_spacing[2]))
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    if resamplemethod == sitk.sitkNearestNeighbor:
        resample.SetOutputPixelType(sitk.sitkUInt16)  # 近邻插值用于mask的，保存uint16
    else:
        resample.SetOutputPixelType(sitk.sitkFloat32)  # 线性插值用于PET/CT/MRI之类的，保存float32
    resample.SetInterpolator(resamplemethod)

    return resample.Execute(itk_image)

def interplote(points):
    added = []
    for i in range(len(points)-1):
        dist = np.linalg.norm(np.array(points[i+1]) - np.array(points[i]))
        if dist > 1.4:
            pair = [points[i], points[i+1]]

            if np.abs(points[i][0]-points[i+1][0]) > np.abs(points[i][1]-points[i+1][1]):

                min_idx = np.argmin([points[i][0], points[i+1][0]])
                xx = np.linspace(start=pair[min_idx][0], stop=pair[1-min_idx]
                                 [0], num=pair[1-min_idx][0]-pair[min_idx][0]+2, dtype='int32')
                interp = np.interp(
                    xx, [pair[min_idx][0], pair[1-min_idx][0]], [pair[min_idx][1], pair[1-min_idx][1]])
                for dummy in zip(xx, interp):
                    added.append([int(dummy[0]), int(dummy[1])])

            else:
                min_idx = np.argmin([points[i][1], points[i+1][1]])
                yy = np.linspace(start=pair[min_idx][1], stop=pair[1-min_idx]
                                 [1], num=pair[1-min_idx][1]-pair[min_idx][1]+2, dtype='int32')
                interp = np.interp(
                    yy, [pair[min_idx][1], pair[1-min_idx][1]], [pair[min_idx][0], pair[1-min_idx][0]])
                for dummy in zip(interp, yy):
                    added.append([int(dummy[0]), int(dummy[1])])

    return [list(x) for x in set(tuple(x) for x in added+points)]

def get_mask2nii(ct, structure, ROI_dict):
    raw_image_size = ct.GetSize()
    origin = ct.GetOrigin()
    pixel_spacing = ct.GetSpacing()
    direction = ct.GetDirection()

    mask_list = []
    for roi in structure.ROIContourSequence:
        number = roi.ReferencedROINumber
        contexist = hasattr(roi, 'ContourSequence')

        if number in ROI_dict['ids'] and contexist:
            np_mask = np.zeros((raw_image_size[2], raw_image_size[1], raw_image_size[0]))
            np_mask.fill(0)

            for cont in roi.ContourSequence:
                data = {
                    'x': ([cont.ContourData[index] for index in
                           range(0, len(cont.ContourData), 3)] if hasattr(cont, 'ContourData') else None),
                    'y': ([cont.ContourData[index + 1] for index in
                           range(0, len(cont.ContourData), 3)] if hasattr(cont, 'ContourData') else None),
                    'z': ([cont.ContourData[index + 2] for index in
                           range(0, len(cont.ContourData), 3)] if hasattr(cont, 'ContourData') else None)
                }
                pts = np.zeros([len(data['x']), 3])
                for index in range(len(data['x'])):
                    world_coords = ct.TransformPhysicalPointToContinuousIndex(
                        (data['x'][index], data['y'][index], data['z'][index]))
                    pts[index, 0] = world_coords[0]
                    pts[index, 1] = world_coords[1]
                    pts[index, 2] = world_coords[2]
                z = int(round(pts[0, 2]))

                try:
                    mask_z = draw.polygon2mask((raw_image_size[1], raw_image_size[0]), np.column_stack((pts[:, 1], pts[:, 0])))
                    np_mask[z, mask_z] = 1
                except IndexError:
                    # if this is triggered the contour is out of bounds
                    raise ContourOutOfBoundsException()
                except RuntimeError as e:
                    # this error is sometimes thrown by SimpleITK if the index goes out of bounds
                    if 'index out of bounds' in str(e):
                        raise ContourOutOfBoundsException()
                    raise e  # something serious is going on

            mask = sitk.GetImageFromArray(np_mask)
            mask.SetSpacing(pixel_spacing)
            mask.SetDirection(direction)
            mask.SetOrigin(origin)
            mask_inf = {'mask': mask,
                        'Sclass': ROI_dict['Sclass'][ROI_dict['ids'].index(number)]}
            #'name': ROI_dict['names'][ROI_dict['ids'].index(number)],
            mask_list.append(mask_inf)
            print('struct {} load'.format(ROI_dict['names'][ROI_dict['ids'].index(number)]))
    return mask_list

def RS2nii(ct_path, mask_folder_path, mask_key):
    roi_dict = []
    roinum_dict = []

    raw_image = sitk.ReadImage(ct_path)
    raw_image_size = raw_image.GetSize()
    origin = raw_image.GetOrigin()
    pixel_spacing = raw_image.GetSpacing()
    direction = raw_image.GetDirection()

    structure_set_file = glob.glob(os.path.join(
        mask_folder_path, '*.dcm'))
    structure = pydicom.read_file(structure_set_file[0], force=True)

    mask_list = []
    for item in structure.StructureSetROISequence:  # 查找带有关键词的的mask
        flag = False
        for key in mask_key:
            if key in item.ROIName.lower():
                flag = True
                continue
        if flag:
            roi_dict.append(item.ROIName)
            roinum_dict.append(item.ROINumber)

    for roi in structure.ROIContourSequence:
        number = roi.ReferencedROINumber
        contexist = hasattr(roi, 'ContourSequence')

        if number in roinum_dict and contexist:
            np_mask = np.zeros((raw_image_size[2], raw_image_size[1], raw_image_size[0]))
            np_mask.fill(0)

            for cont in roi.ContourSequence:
                data = {
                    'x': ([cont.ContourData[index] for index in
                           range(0, len(cont.ContourData), 3)] if hasattr(cont, 'ContourData') else None),
                    'y': ([cont.ContourData[index + 1] for index in
                           range(0, len(cont.ContourData), 3)] if hasattr(cont, 'ContourData') else None),
                    'z': ([cont.ContourData[index + 2] for index in
                           range(0, len(cont.ContourData), 3)] if hasattr(cont, 'ContourData') else None)
                }
                pts = np.zeros([len(data['x']), 3])
                for index in range(len(data['x'])):
                    world_coords = raw_image.TransformPhysicalPointToContinuousIndex(
                        (data['x'][index], data['y'][index], data['z'][index]))
                    pts[index, 0] = world_coords[0]
                    pts[index, 1] = world_coords[1]
                    pts[index, 2] = world_coords[2]
                z = int(round(pts[0, 2]))

                try:
                    mask_z = draw.polygon2mask((raw_image_size[1], raw_image_size[0]), np.column_stack((pts[:, 1], pts[:, 0])))
                    np_mask[z, mask_z] = 1
                except IndexError:
                    # if this is triggered the contour is out of bounds
                    raise ContourOutOfBoundsException()
                except RuntimeError as e:
                    # this error is sometimes thrown by SimpleITK if the index goes out of bounds
                    if 'index out of bounds' in str(e):
                        raise ContourOutOfBoundsException()
                    raise e  # something serious is going on

            mask = sitk.GetImageFromArray(np_mask)
            mask.SetSpacing(pixel_spacing)
            mask.SetDirection(direction)
            mask.SetOrigin(origin)
            mask_list.append(mask)
            print('struct {} load'.format(roi_dict[roinum_dict.index(number)]))
    return mask_list


def check_shape_and_resize(dose, image):
    if image.GetSize() != dose.GetSize() or \
            image.GetOrigin() != dose.GetOrigin() or \
            image.GetSpacing() != dose.GetSpacing():
        resize_dose = resize_image_itk(dose, image, sitk.sitkLinear)
        print('====>Dose resized!')
        print("     DOSE resized spacing:", resize_dose.GetSpacing())
        print("     DOSE resized size:", resize_dose.GetSize())
        print("     DOSE resized origin:", resize_dose.GetOrigin())
    else:
        resize_dose = dose
    return resize_dose


def crop_img(img, center_ind, new_size):
    image_arr = sitk.GetArrayFromImage(img)
    ori_size = image_arr.shape

    star_ind = [int(np.max([center_ind[0] - new_size[0] / 2, 0])),
                int(np.max([center_ind[1] - new_size[1] / 2, 0])),
                int(np.max([center_ind[2] - new_size[2] / 2, 0]))]
    end_ind = [int(np.min([star_ind[0] + new_size[0], ori_size[0]])),
               int(np.min([star_ind[1] + new_size[1], ori_size[1]])),
               int(np.min([star_ind[2] + new_size[2], ori_size[2]]))]
    crop_leng = [int(end_ind[0] - star_ind[0]),
                 int(end_ind[1] - star_ind[1]),
                 int(end_ind[2] - star_ind[2])]

    new_image_arr = np.zeros(new_size)
    new_image_arr.fill(image_arr[0, 0, 0])

    new_image_arr[:crop_leng[0], :crop_leng[1], :crop_leng[2]] = \
        image_arr[star_ind[0]:end_ind[0], star_ind[1]:end_ind[1], star_ind[2]:end_ind[2]]

    new_img = sitk.GetImageFromArray(new_image_arr)
    new_img.SetSpacing(img.GetSpacing())
    new_img.SetDirection(img.GetDirection())
    oridata = img.TransformContinuousIndexToPhysicalPoint((star_ind[2], star_ind[1], star_ind[0]))
    new_img.SetOrigin(oridata)
    return new_img
