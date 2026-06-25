# -*- coding: utf-8 -*-
from radiomics import featureextractor
import os
from os.path import join

import numpy as np
import pandas as pd

base_path = r'D:\project\0_Competition\data\new_data\CT_dose_resample_333'
# Instantiate the extractor
paramPath_ct = r'D:\code_updata\Dual_SEnet\Radiomics\params_ct.yaml'
extractor_ct = featureextractor.RadiomicsFeatureExtractor(paramPath_ct)
paramPath_dose = r'D:\code_updata\Dual_SEnet\Radiomics\params_dose.yaml'
extractor_dose = featureextractor.RadiomicsFeatureExtractor(paramPath_dose)
radiomics_value = None
radiomics_key = None

patient_list = os.listdir(base_path)
for patient in patient_list:
    print(patient + ' data Loading...')
    ct_path = join(base_path, patient, 'CT.nii.gz')
    dose_path = join(base_path, patient, 'DOSE.nii.gz')
    mask_path = join(base_path, patient, 'GTV_mask.nii.gz')

    radiomics_ct = extractor_ct.execute(ct_path, mask_path)
    radiomics_ct_list = list(radiomics_ct.values())
    radiomics_ct_list = radiomics_ct_list[37:]

    radiomics_dose = extractor_dose.execute(dose_path, mask_path)
    radiomics_dose_list = list(radiomics_dose.values())
    radiomics_dose_list = radiomics_dose_list[37:]

    radiomics_np = np.vstack(radiomics_ct_list + radiomics_dose_list)

    if radiomics_value is None:
        radiomics_value = radiomics_np
    else:
        radiomics_value = np.hstack((radiomics_value, radiomics_np))

    if radiomics_key is None:
        ct_label = list(radiomics_ct.keys())
        ct_label = list(map(lambda x: 'CT_' + x, ct_label[37:]))
        dose_label = list(radiomics_dose.keys())
        dose_label = list(map(lambda x: 'Dose_' + x, dose_label[37:]))
        radiomics_key = ct_label + dose_label

radiomics_all = np.transpose(radiomics_value)
radiomics_pd = pd.DataFrame(radiomics_all, columns=radiomics_key, index=patient_list)
radiomics_pd.to_excel(r'D:\project\0_Competition\data\new_data\radiomics_0828.xlsx')
