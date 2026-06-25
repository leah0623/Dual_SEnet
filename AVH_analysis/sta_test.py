import numpy as np
from statsmodels.stats.multitest import multipletests
from scipy import stats
import pandas as pd

base_path = '../Deeplearning/experiments/Dual_task_CD/gradcam_cal_DM.xlsx'
baserit = pd.read_excel(base_path)
traindata = baserit[baserit['dataset'] == 'train']
validdata = baserit[baserit['dataset'] == 'valid']
testdata = baserit[baserit['dataset'] == 'test']
cut_flag = 'groups'

colnamelist = np.array(baserit.columns[9:])

def statiscTest(pddata, groupname, colnamelist):
    highidx = np.array(pddata[groupname]) == 1
    lowidx = np.array(pddata[groupname]) == 0
    pvalue = []
    for i in range(len(colnamelist)):
        data1 = np.array(pddata[colnamelist[i]][highidx])
        data2 = np.array(pddata[colnamelist[i]][lowidx])
        u_stat, p_value = stats.mannwhitneyu(data1, data2)

        splitname = colnamelist[i].split('_')
        savedata = {'GTV': splitname[0], 'dose': splitname[1], 'threshold': splitname[2], 'pvalue': p_value}
        pvalue.append(savedata)
    pvalue = pd.DataFrame(pvalue)
    return pvalue

def fdr(pvalue_ext):
    ROI_list = ['G0', 'G3', 'G03']
    dose_list = ['D40', 'D45', 'D50', 'D55', 'D60', 'D65', 'D70']
    fdr_check = pvalue_ext.copy()
    fdr_check.insert(4, 'p_fdr', 0)
    for roi in ROI_list:
        for dose in dose_list:
            group_index = (fdr_check['GTV'] == roi) & (fdr_check['dose'] == dose)
            pvalues = fdr_check[group_index]['pvalue'].values
            significant, p_fdr, _, _ = multipletests(pvalues, alpha=0.05, method='fdr_bh')
            fdr_check.iloc[np.where(group_index)[0], 4] = p_fdr
    return fdr_check

def fdr_all(pvalue_data):
    fdr_check = pvalue_data.copy()
    significant, p_fdr, _, _ = multipletests(fdr_check['pvalue'].values, alpha=0.05, method='fdr_bh')
    fdr_check.insert(4, 'p_fdr', p_fdr)
    return fdr_check

pvalue_tra = statiscTest(traindata, cut_flag, colnamelist)
pvalue_tra_check = fdr_all(pvalue_tra)

pvalue_int = statiscTest(validdata, cut_flag, colnamelist)
pvalue_int_check = fdr_all(pvalue_int)

pvalue_ext = statiscTest(testdata, cut_flag, colnamelist)
pvalue_ext_check = fdr_all(pvalue_ext)

with pd.ExcelWriter('../Deeplearning/experiments/Dual_task_CD/cam_mu_test_DM.xlsx') as writer:
    pvalue_tra_check.to_excel(writer, sheet_name='Train', index=False)
    pvalue_int_check.to_excel(writer, sheet_name='Internal', index=False)
    pvalue_ext_check.to_excel(writer, sheet_name='External', index=False)


