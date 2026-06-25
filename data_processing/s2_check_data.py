import os
from os.path import join
import glob
import numpy as np
import pydicom

base_data_dir = r'D:\project\0_Competition\data\DCM_data'
patient_list = os.listdir(base_data_dir)
save_dir = r'D:\project\0_Competition\data\processd\Data_nii'

for name in patient_list:
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

    if len(dose_data) != 1:
        print('----------- {} -----------'.format(name))
        print('These files match the struct:')
        get_strUID = np.zeros(len(dose_data))
        for dose_file in dose_data:
            dose_dcm = pydicom.read_file(dose_file)
            if hasattr(dose_dcm, 'ReferencedStructureSetSequence'):
                if dose_dcm.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID == str_dcm.SOPInstanceUID:
                    print(dose_file.split('\\')[-1])
            elif hasattr(dose_dcm, 'ReferencedRTPlanSequence'):
                if dose_dcm.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID in march_plan:
                    print(dose_file.split('\\')[-1])