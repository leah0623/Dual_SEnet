import torch
from torch.utils.data import Dataset
import torchio as tio
import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored

import os
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from Deeplearning.dataloader.read_data import read_follow_data, read_data, get_transform
from Deeplearning.models.dual_task_model import generate_parallel_model
from Deeplearning.utils.utilis_code import find_optimal_cutoff



plt.switch_backend('agg')

def time_dependent_auc(hazards, event_times, event_indicators, time_point):
    """
    计算特定时间点的AUC (Area Under Curve)

    参数:
    hazards: 模型预测的风险值 (n, 1)
    event_times: 实际生存时间 (n,)
    event_indicators: 事件指示器 (n,)
    time_point: 评估的时间点

    返回:
    AUC值
    """
    hazards = hazards.reshape(-1)
    # 筛选出在时间点前发生事件的样本
    event_mask = (event_times <= time_point) & (event_indicators == 1)
    # 筛选出在时间点后仍存活的样本
    survival_mask = event_times > time_point

    if np.sum(event_mask) == 0 or np.sum(survival_mask) == 0:
        return 0.5  # 无法计算时返回随机猜测值

    # 提取相关样本
    event_risks = hazards[event_mask]
    survival_risks = hazards[survival_mask]

    # 计算AUC
    auc = 0
    for er in event_risks:
        for sr in survival_risks:
            if er > sr:
                auc += 1
    auc /= (len(event_risks) * len(survival_risks))

    return auc

def get_evaluation_metrics(hazards, event_times, event_indicators):
    cindex = concordance_index_censored(event_indicators.squeeze().astype(bool), event_times.squeeze(), hazards.squeeze())[0]
    auc_1y = time_dependent_auc(hazards, event_times, event_indicators, 365 * 1)
    auc_3y = time_dependent_auc(hazards, event_times, event_indicators, 365 * 3)
    auc_5y = time_dependent_auc(hazards, event_times, event_indicators, 365 * 5)

    return cindex, auc_1y, auc_3y, auc_5y

def data_into_input(tiodata, device):
    targets_DM = tiodata['target_DM'].long()
    targets_LR = tiodata['target_LR'].long()
    targets_DM = targets_DM.to(device, non_blocking=True)
    targets_LR = targets_LR.to(device, non_blocking=True)

    times_DM = tiodata['event_time_DM'].long()
    times_LR = tiodata['event_time_LR'].long()
    times_DM = times_DM.to(device, non_blocking=True)
    times_LR = times_LR.to(device, non_blocking=True)

    targets = {'DM': targets_DM, 'DM_times': times_DM,
               'LR': targets_LR, 'LR_times': times_LR}
    cts = tiodata['ct'][tio.DATA]
    cts = cts.permute(0, 1, 4, 2, 3)
    masks = tiodata['mask'][tio.DATA]
    masks = masks.permute(0, 1, 4, 2, 3)
    doses = tiodata['dose'][tio.DATA]
    doses = doses.permute(0, 1, 4, 2, 3)
    inputs1 = torch.cat((cts, masks), dim=1)
    inputs1 = inputs1.to(device, non_blocking=True)
    inputs2 = torch.cat((doses, masks), dim=1)
    inputs2 = inputs2.to(device, non_blocking=True)
    inputs = [inputs1, inputs2]
    return inputs, targets

def run_test(data_subject, device, model):
    dataset = tio.SubjectsDataset(data_subject, transform=get_transform())
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                              num_workers=0, pin_memory=True)
    pred_DM_risk = []
    true_DM_events = []
    true_DM_times = []

    pred_LR_risk = []
    true_LR_events = []
    true_LR_times = []

    data_case_name = []

    model.eval()
    with torch.no_grad():
        for i, subjects_batch in enumerate(data_loader):
            name = subjects_batch['name']
            data_case_name.append(name)
            inputs, targets = data_into_input(subjects_batch, device)
            DMout, LRout = model(inputs)

            pred_DM_risk.append(DMout.cpu().detach().numpy())
            true_DM_events.append(targets['DM'].cpu().detach().numpy())
            true_DM_times.append(targets['DM_times'].cpu().detach().numpy())

            pred_LR_risk.append(LRout.cpu().detach().numpy())
            true_LR_events.append(targets['LR'].cpu().detach().numpy())
            true_LR_times.append(targets['LR_times'].cpu().detach().numpy())

    pred_DM_risk = np.concatenate(pred_DM_risk)[:, 0]
    true_DM_events = np.concatenate(true_DM_events)
    true_DM_times = np.concatenate(true_DM_times)

    pred_LR_risk = np.concatenate(pred_LR_risk)[:, 0]
    true_LR_events = np.concatenate(true_LR_events)
    true_LR_times = np.concatenate(true_LR_times)

    torch.cuda.empty_cache()
    data_DM = {'casename': np.concatenate(data_case_name),
               'risk_score': pred_DM_risk,
               'event_time': true_DM_times,
               'event_indicators': true_DM_events}
    data_LR = {'casename': np.concatenate(data_case_name),
               'risk_score': pred_LR_risk,
               'event_time': true_LR_times,
               'event_indicators': true_LR_events}
    return data_DM, data_LR

def cal_metrics(name, hazards, event_times, event_indicators, cutoff=None):
    cindex = concordance_index_censored(event_indicators.squeeze().astype(bool), event_times.squeeze(), hazards.squeeze())[0]
    auc_1y = time_dependent_auc(hazards, event_times, event_indicators, 365 * 1)
    auc_3y = time_dependent_auc(hazards, event_times, event_indicators, 365 * 3)
    auc_5y = time_dependent_auc(hazards, event_times, event_indicators, 365 * 5)
    if cutoff is None:
        fpr, tpr, thresholds = roc_curve(event_indicators, hazards)
        cutoff = find_optimal_cutoff(tpr, fpr, thresholds)

    auc = roc_auc_score(event_indicators, hazards)
    pred_indicators = list(map(lambda x: 1 if x >= cutoff else 0, hazards))
    TN, FP, FN, TP = confusion_matrix(event_indicators, pred_indicators).ravel()
    specificity = TN / (TN + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)

    print('----------------------- {} --------------------'.format(name))
    print('the best cutoff is {:.4f}'.format(cutoff))
    print('the AUC is {:.4f}'.format(auc))
    print('the Specificity is {:.4f}'.format(specificity))
    print('the Sensitivity(Recall) is {:.4f}'.format(recall))
    print('the C-index is {:.4f}'.format(cindex))
    print('the Time-AUCs at 1,3,5y are {:.4f}, {:.4f}, {:.4f}'.format(auc_1y, auc_3y, auc_5y))

    return [auc, specificity, recall, cindex, auc_1y, auc_3y, auc_5y, cutoff]

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定使用哪一块gpu

    input_mode = 'CT_DOSE'
    save_dir = os.path.join('./experiments', 'Dual_task_CD')
    model_name = 'inter_best_cindex_model'
    path_checkpoint = os.path.join(save_dir, '{}.pth'.format(model_name))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir =r'E:\LHL\competition\CJJ\Data\CT_RD_nii_resample222'
    data_outcome_dir = r'E:\LHL\competition\CJJ\Data\ALL_dm_lr.xlsx'
    save_path = os.path.join(save_dir, 'result_{}.xlsx'.format(model_name))

    HGJ_name, HGJ_event_DM, HGJ_time_DM, HGJ_event_LR, HGJ_time_LR = read_follow_data(data_outcome_dir, 'HGJ')
    CHUS_name, CHUS_event_DM, CHUS_time_DM, CHUS_event_LR, CHUS_time_LR = read_follow_data(data_outcome_dir, 'CHUS')
    HMR_name, HMR_event_DM, HMR_time_DM, HMR_event_LR, HMR_time_LR = read_follow_data(data_outcome_dir, 'HMR')
    CHUM_name, CHUM_event_DM, CHUM_time_DM, CHUM_event_LR, CHUM_time_LR = read_follow_data(data_outcome_dir, 'CHUM')

    train_subject = read_data(
        data_dir, np.concatenate((HGJ_name, CHUS_name)),
        np.concatenate((HGJ_event_DM, CHUS_event_DM)),
        np.concatenate((HGJ_time_DM, CHUS_time_DM)),
        np.concatenate((HGJ_event_LR, CHUS_event_LR)),
        np.concatenate((HGJ_time_LR, CHUS_time_LR)))

    inter_subject = read_data(data_dir, CHUM_name, CHUM_event_DM, CHUM_time_DM, CHUM_event_LR, CHUM_time_LR)
    exter_subject = read_data(data_dir, HMR_name, HMR_event_DM, HMR_time_DM, HMR_event_LR, HMR_time_LR)

    model = generate_parallel_model(n_input_channels=2, num_classes=1)
    checkpoint = torch.load(path_checkpoint)
    model.load_state_dict(checkpoint['net'])
    model = model.to(device)

    train_data_DM, train_data_LR = run_test(train_subject, device, model)
    train_DM = cal_metrics('train DM', train_data_DM['risk_score'], train_data_DM['event_time'], train_data_DM['event_indicators'])
    train_LR = cal_metrics('train LR', train_data_LR['risk_score'], train_data_LR['event_time'], train_data_LR['event_indicators'])

    inter_data_DM, inter_data_LR = run_test(inter_subject, device, model)
    inter_DM = cal_metrics('inter DM', inter_data_DM['risk_score'], inter_data_DM['event_time'], inter_data_DM['event_indicators'])
    inter_LR = cal_metrics('inter LR', inter_data_LR['risk_score'], inter_data_LR['event_time'], inter_data_LR['event_indicators'])

    exter_data_DM, exter_data_LR = run_test(exter_subject, device, model)
    exter_DM = cal_metrics('exter DM', exter_data_DM['risk_score'], exter_data_DM['event_time'], exter_data_DM['event_indicators'])
    exter_LR = cal_metrics('exter LR', exter_data_LR['risk_score'], exter_data_LR['event_time'], exter_data_LR['event_indicators'])

    # write data
    train_DM_pd = pd.DataFrame(train_data_DM)
    inter_DM_pd = pd.DataFrame(inter_data_DM)
    exter_DM_pd = pd.DataFrame(exter_data_DM)
    train_LR_pd = pd.DataFrame(train_data_LR)
    inter_LR_pd = pd.DataFrame(inter_data_LR)
    exter_LR_pd = pd.DataFrame(exter_data_LR)
    meritics = {'train_DM': train_DM, 'inter_DM': inter_DM,'exter_DM': exter_DM,
                'train_LR': train_LR, 'inter_LR': inter_LR, 'exter_LR': exter_LR}
    meritics_pd = pd.DataFrame(meritics, index=['AUC', 'Apecificity', 'Sensitivity',
                                                'Cindex', 'auc_1y', 'auc_3y', 'auc_5y', 'cutoff'])
    with pd.ExcelWriter(save_path) as writer:
        train_DM_pd.to_excel(writer, sheet_name='DMtrain', index=False)
        inter_DM_pd.to_excel(writer, sheet_name='DMinter', index=False)
        exter_DM_pd.to_excel(writer, sheet_name='DMexter', index=False)
        train_LR_pd.to_excel(writer, sheet_name='LRtrain', index=False)
        inter_LR_pd.to_excel(writer, sheet_name='LRinter', index=False)
        exter_LR_pd.to_excel(writer, sheet_name='LRexter', index=False)
        meritics_pd.to_excel(writer, sheet_name='meritics')


