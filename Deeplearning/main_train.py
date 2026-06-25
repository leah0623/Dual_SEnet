import torch
from torch.utils.data import Dataset
import torchio as tio
import numpy as np
import pandas as pd
from itertools import chain

from sksurv.metrics import concordance_index_censored

import os
import argparse
from matplotlib import pyplot as plt
import time
from prettytable import PrettyTable

from Deeplearning.utils.base_net import set_all_seeds, Trainsave
from Deeplearning.utils.utilis_code import create_lr_scheduler
from Deeplearning.utils.visualizer import Visualizer
from Deeplearning.dataloader.read_data import read_follow_data, read_data, get_weight, get_transform
from Deeplearning.models.dual_task_model import generate_parallel_model
from Deeplearning.loss.survialloss import weighted_surv_loss

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

def training_epoch(
        model,
        train_loader,
        optimizer,
        lr_scheduler,
        device,
        lossfunction,
        if_print_details=False,
        updata_each_item=True):
    model.train()
    optimizer.zero_grad()

    train_loss = 0
    train_loss_DM = 0
    train_loss_LR = 0

    pred_DM_risk = []
    true_DM_events = []
    true_DM_times = []

    pred_LR_risk = []
    true_LR_events = []
    true_LR_times = []

    item = 0
    accumulation_steps = 2

    for i, subjects_batch in enumerate(train_loader):
        inputs, targets = data_into_input(subjects_batch, device)
        DMout, LRout = model(inputs)

        DMloss = lossfunction(DMout, targets['DM_times'], targets['DM'])
        LRloss = lossfunction(LRout, targets['LR_times'], targets['LR'])

        loss = (DMloss + LRloss) / accumulation_steps
        loss.backward()

        if (i+1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            item += 1

            if updata_each_item:
                lr_scheduler.step()

        train_loss += loss.item()
        train_loss_DM += DMloss.item() / accumulation_steps
        train_loss_LR += LRloss.item() / accumulation_steps

        pred_DM_risk.append(DMout.cpu().detach().numpy())
        true_DM_events.append(targets['DM'].cpu().detach().numpy())
        true_DM_times.append(targets['DM_times'].cpu().detach().numpy())

        pred_LR_risk.append(LRout.cpu().detach().numpy())
        true_LR_events.append(targets['LR'].cpu().detach().numpy())
        true_LR_times.append(targets['LR_times'].cpu().detach().numpy())

    # cal evaluation
    pred_DM_risk = np.concatenate(pred_DM_risk)
    true_DM_events = np.concatenate(true_DM_events)
    true_DM_times = np.concatenate(true_DM_times)
    DMcindex, DMauc_1y, DMauc_3y, DMauc_5y = get_evaluation_metrics(pred_DM_risk, true_DM_times, true_DM_events)

    pred_LR_risk = np.concatenate(pred_LR_risk)
    true_LR_events = np.concatenate(true_LR_events)
    true_LR_times = np.concatenate(true_LR_times)
    LRcindex, LRauc_1y, LRauc_3y, LRauc_5y = get_evaluation_metrics(pred_LR_risk, true_LR_times, true_LR_events)

    if if_print_details:
        tb = PrettyTable(["DM_label", "DM_risk", "LR_label", 'LR_risk'])
        for i in range(len(pred_DM_risk)):
            tb.add_row([true_DM_events[i], '{:.3f}'.format(pred_DM_risk[i][0]), true_LR_events[i], '{:.3f}'.format(pred_LR_risk[i][0])])
        print(tb)

    DMresult = {'loss': train_loss_DM / item, 'Cindex': DMcindex, 'TimeAUC': (DMauc_1y + DMauc_3y + DMauc_5y)/3}
    LRresult = {'loss': train_loss_LR / item, 'Cindex': LRcindex, 'TimeAUC': (LRauc_1y + LRauc_3y + LRauc_5y) / 3}

    return train_loss / item, DMresult, LRresult

def validation_epoch(
        model,
        valid_loader,
        device,
        lossfunction,
        if_print_details=False):
    model.eval()
    valid_loss = 0
    valid_loss_DM = 0
    valid_loss_LR = 0

    pred_DM_risk = []
    true_DM_events = []
    true_DM_times = []

    pred_LR_risk = []
    true_LR_events = []
    true_LR_times = []

    item = 0
    with torch.no_grad():
        for i, subjects_batch in enumerate(valid_loader):
            inputs, targets = data_into_input(subjects_batch, device)
            DMout, LRout = model(inputs)

            DMloss = lossfunction(DMout, targets['DM_times'], targets['DM'])
            LRloss = lossfunction(LRout, targets['LR_times'], targets['LR'])

            valid_loss += (DMloss + LRloss).item()
            valid_loss_DM += DMloss.item()
            valid_loss_LR += LRloss.item()
            item += 1

            pred_DM_risk.append(DMout.cpu().detach().numpy())
            true_DM_events.append(targets['DM'].cpu().detach().numpy())
            true_DM_times.append(targets['DM_times'].cpu().detach().numpy())

            pred_LR_risk.append(LRout.cpu().detach().numpy())
            true_LR_events.append(targets['LR'].cpu().detach().numpy())
            true_LR_times.append(targets['LR_times'].cpu().detach().numpy())

    pred_DM_risk = np.concatenate(pred_DM_risk)
    true_DM_events = np.concatenate(true_DM_events)
    true_DM_times = np.concatenate(true_DM_times)
    DMcindex, DMauc_1y, DMauc_3y, DMauc_5y = get_evaluation_metrics(pred_DM_risk, true_DM_times, true_DM_events)

    pred_LR_risk = np.concatenate(pred_LR_risk)
    true_LR_events = np.concatenate(true_LR_events)
    true_LR_times = np.concatenate(true_LR_times)
    LRcindex, LRauc_1y, LRauc_3y, LRauc_5y = get_evaluation_metrics(pred_LR_risk, true_LR_times, true_LR_events)

    DMresult = {'loss': valid_loss_DM / item, 'Cindex': DMcindex, 'TimeAUC': (DMauc_1y + DMauc_3y + DMauc_5y) / 3}
    LRresult = {'loss': valid_loss_LR / item, 'Cindex': LRcindex, 'TimeAUC': (LRauc_1y + LRauc_3y + LRauc_5y) / 3}

    if if_print_details:
        tb = PrettyTable(["DM_label", "DM_risk", "LR_label", 'LR_risk'])
        for i in range(len(pred_DM_risk)):
            tb.add_row([true_DM_events[i], '{:.3f}'.format(pred_DM_risk[i][0]), true_LR_events[i], '{:.3f}'.format(pred_LR_risk[i][0])])
        print(tb)

    return valid_loss / item, DMresult, LRresult

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyper-parameters management')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train set')
    parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train (default: 300)')

    # Dataset parameters
    parser.add_argument('--balanced', type=bool, default=False, help='Dataset is balanced')
    parser.add_argument('--input_mode', default='CT_DOSE', type=str, help='Input mode (default:"CT/DOSE/CT_DOSE")')
    parser.add_argument('--num_classes', type=int, default=1, help='classes num')
    parser.add_argument('--resume', type=bool, default=False, help='resume training')
    parser.add_argument('--pretrained', type=bool, default=False, help='load pretrained model for training')
    parser.add_argument('--save_name', type=str, default='Dual_task_CD', help='save path of trained model')
    parser.add_argument('--image_dir', type=str,
                        default=r'E:\LHL\competition\CJJ\Data\CT_RD_nii_resample333',
                        help='nii images dir')
    parser.add_argument('--flag_path', type=str,
                        default=r'E:\LHL\competition\CJJ\Data\ALL_dm_lr.xlsx',
                        help='flag(time-to-event data)')

    # Model parameters
    parser.add_argument('--model_name', default='dual_task', type=str, help='Model name(default:"resnet/densenet/resnext"')
    parser.add_argument('--resume_path', type=str, default=None, help='resume path of trained model')
    parser.add_argument('--resume', type=bool, default=False, help='if resume')

    # Optimization parameters
    parser.add_argument('--opt', default='adamw', type=str, help='Optimizer (default: "sgd"')
    parser.add_argument('--momentum', type=float, default=0.9, help='Optimizer momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=2e-5, help='weight decay (default: 2e-5)') # 1e-3
    parser.add_argument('--lr', type=float, default=1e-6, help='learning rate (default: 0.05)')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='learning rate (default: 0.05)')
    parser.add_argument('--warmup_lr', type=float, default=0, help='learning rate (default: 0.05)')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='number of epochs to warmup (default: 5)')
    parser.add_argument('--scheduler_name', default='cosine', type=str, help='Scheduler_name (default: "cosine"')
    parser.add_argument('--scheduler_decay_epochs', type=int, default=10, help='number of epochs to warmup (default: 5)')
    parser.add_argument('--scheduler_decay_rate', type=float, default=0.1, help='learning rate (default: 0.05)')

    # Mixup parameters
    parser.add_argument('--mixup', type=float, default=0.2, help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0., help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--smoothing', type=float, default=0.2, help='Label smoothing (default: 0.1)')

    args = parser.parse_args(args=[])

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_all_seeds(args.seed)  # seeds

    # load data
    HGJ_name, HGJ_event_DM, HGJ_time_DM, HGJ_event_LR, HGJ_time_LR = read_follow_data(args.flag_path, 'HGJ')
    CHUS_name, CHUS_event_DM, CHUS_time_DM, CHUS_event_LR, CHUS_time_LR = read_follow_data(args.flag_path, 'CHUS')
    HMR_name, HMR_event_DM, HMR_time_DM, HMR_event_LR, HMR_time_LR = read_follow_data(args.flag_path, 'HMR')
    CHUM_name, CHUM_event_DM, CHUM_time_DM, CHUM_event_LR, CHUM_time_LR = read_follow_data(args.flag_path, 'CHUM')

    train_subject = read_data(
        args.image_dir, np.concatenate((HGJ_name, CHUS_name)),
        np.concatenate((HGJ_event_DM, CHUS_event_DM)),
        np.concatenate((HGJ_time_DM, CHUS_time_DM)),
        np.concatenate((HGJ_event_LR, CHUS_event_LR)),
        np.concatenate((HGJ_time_LR, CHUS_time_LR)))
    train_dataset = tio.SubjectsDataset(train_subject, transform=get_transform(mode='train'))
    if not args.balanced:
        train_event = np.concatenate((HGJ_event_DM, CHUS_event_DM)) + np.concatenate((HGJ_event_LR, CHUS_event_LR))
        train_weight = get_weight(train_event)
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weight, len(train_dataset),
                                                                       replacement=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                                                   num_workers=0, pin_memory=True, drop_last=True)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   shuffle=True, num_workers=0,
                                                   pin_memory=True, drop_last=True)

    inter_subject = read_data(args.image_dir, CHUM_name, CHUM_event_DM, CHUM_time_DM, CHUM_event_LR, CHUM_time_LR)
    inter_dataset = tio.SubjectsDataset(inter_subject, transform=get_transform(mode='valid'))
    inter_loader = torch.utils.data.DataLoader(inter_dataset, batch_size=args.batch_size,
                                               shuffle=False, num_workers=0,
                                               pin_memory=True, drop_last=False)

    # set path
    save_path = os.path.join('./experiments', args.save_name)
    os.makedirs(save_path, exist_ok=True)
    save_log_path = os.path.join(save_path, 'log.xlsx')

    # load model
    model = generate_parallel_model(n_input_channels=2, num_classes=args.num_classes)
    viz = Visualizer(args.save_path)
    epoch = 0
    if args.resume:
        path_checkpoint = "./experiments/" + args.resume_path + "/val_loss_minimum_model.pth"
        checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(checkpoint['net'])
        epoch = checkpoint['epoch']
    model = model.to(device)

    # set diffrent learn rate
    params_group1 = chain(model.layer0_1.parameters(), model.layer0_2.parameters(),
                          model.layer1.parameters(), model.layer2.parameters())
    params_group2 = chain(model.layer3_DM.parameters(), model.layer4_DM.parameters(),
                          model.class_DM.parameters())
    params_group3 = chain(model.layer3_LR.parameters(), model.layer4_LR.parameters(),
                          model.class_LR.parameters())
    optimizer = torch.optim.AdamW([
        {'params': params_group1, 'lr': args.lr * 4},
        {'params': params_group2, 'lr': args.lr * 2},
        {'params': params_group3, 'lr': args.lr * 40}
    ], weight_decay=args.weight_decay)
    scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                    warmup=True, warmup_epochs=args.warmup_epochs,
                                    warmup_factor=1e-3, end_factor=0.1)

    # set loss function
    lossfunction = weighted_surv_loss()

    # set save log
    alltrain = Trainsave()
    allvalid = Trainsave()
    DMtrain = Trainsave()
    DMvalid = Trainsave()
    LRtrain = Trainsave()
    LRvalid = Trainsave()

    # star train
    print('Start Training')
    print('-' * 30)
    start_time = time.time()
    save_alldata = []
    epochs = []
    early_stop = False
    early_valid = 9999
    early_epoch = 0

    while (epoch < args.epochs + 1) and (not early_stop):
        epoch_time_start = time.time()
        epoch = epoch + 1
        epochs.append(epoch)
        if epoch % 10 == 0 or epoch == 1:
            if_print = True
        else:
            if_print = False

        train_loss, trainDMresult, trainLRresult = training_epoch(model, train_loader, optimizer, scheduler, device, lossfunction, if_print_details=if_print)
        inter_loss, interDMresult, interLRresult = validation_epoch(model, inter_loader, device, lossfunction, if_print_details=if_print)

        # plot loss and other metriels
        viz.plot_stack({'all': train_loss, 'DM':  trainDMresult['loss'],
                        'LR': trainLRresult['loss']}, win='Train loss')
        viz.plot_stack({'all': inter_loss, 'DM':  interDMresult['loss'],
                        'LR': interLRresult['loss']}, win='Inter loss')
        viz.plot_stack({'train': trainDMresult['Cindex'], 'internal':  interDMresult['Cindex']},
                       win='DM_Cindex')
        viz.plot_stack({'train': trainLRresult['Cindex'], 'internal':  interLRresult['Cindex']},
                       win='LR_Cindex')

        # save results
        epoch_time_end = time.time()
        item_result = {'train_loss': train_loss, 'inter_loss': inter_loss,
                       'train_DMloss': trainDMresult['loss'], 'inter_DMloss': interDMresult['loss'],
                       'train_LRloss': trainLRresult['loss'], 'inter_LRloss': interLRresult['loss'],
                       'train_DMCindex': trainDMresult['Cindex'], 'inter_DMCindex': interDMresult['Cindex'],
                       'train_LRCindex': trainLRresult['Cindex'], 'inter_LRCindex': interLRresult['Cindex'],
                       'train_DMTAUC': trainDMresult['TimeAUC'], 'inter_DMTAUC': interDMresult['TimeAUC'],
                       'train_LRTAUC': trainLRresult['TimeAUC'], 'inter_LRTAUC': interLRresult['TimeAUC']}
        save_alldata.append(item_result)

        # save models
        save_name = []
        # BEST LOSS
        saveitem_name = alltrain.save_loss(train_loss, epoch, item_result)
        if saveitem_name is not None:
            save_name.append('train_{}'.format(saveitem_name))

        saveitem_name = allvalid.save_loss(train_loss, epoch, item_result)
        if saveitem_name is not None:
            save_name.append('inter_{}'.format(saveitem_name))

        saveitem_name = DMtrain.save_loss(item_result['train_DMloss'], epoch, item_result)
        if saveitem_name is not None:
            save_name.append('train_DM_{}'.format(saveitem_name))

        saveitem_name = DMvalid.save_loss(item_result['inter_DMloss'], epoch, item_result)
        if saveitem_name is not None:
            save_name.append('inter_DM_{}'.format(saveitem_name))

        saveitem_name = LRtrain.save_loss(item_result['train_LRloss'], epoch, item_result)
        if saveitem_name is not None:
            save_name.append('train_LR_{}'.format(saveitem_name))

        saveitem_name = LRvalid.save_loss(item_result['inter_LRloss'], epoch, item_result)
        if saveitem_name is not None:
            save_name.append('inter_LR_{}'.format(saveitem_name))

        # BEST CINDEX
        if item_result['train_DMCindex'] > 0.70 and item_result['train_LRCindex'] > 0.70:
            saveitem_name = alltrain.save_cindex(item_result['train_DMCindex'] + item_result['train_LRCindex'],
                                                 epoch, item_result)
            if saveitem_name is not None:
                save_name.append('train_{}'.format(saveitem_name))

            saveitem_name = allvalid.save_cindex(item_result['inter_DMCindex'] + item_result['inter_LRCindex'],
                                                 epoch, item_result)
            if saveitem_name is not None:
                save_name.append('inter_{}'.format(saveitem_name))

            saveitem_name = DMtrain.save_cindex(item_result['train_DMCindex'], epoch, item_result)
            if saveitem_name is not None:
                save_name.append('train_DM_{}'.format(saveitem_name))

            saveitem_name = DMvalid.save_cindex(item_result['inter_DMCindex'], epoch, item_result)
            if saveitem_name is not None:
                save_name.append('inter_DM_{}'.format(saveitem_name))

            saveitem_name = LRtrain.save_cindex(item_result['train_LRCindex'], epoch, item_result)
            if saveitem_name is not None:
                save_name.append('train_LR_{}'.format(saveitem_name))

            saveitem_name = LRvalid.save_cindex(item_result['inter_LRCindex'], epoch, item_result)
            if saveitem_name is not None:
                save_name.append('inter_LR_{}'.format(saveitem_name))

        save_name.append('laster_model.pth')
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args}
        for name in save_name:
            torch.save(state, os.path.join(save_path, name))

        # print results
        print("Epoch {}: lr: {:}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
        print("---> Loss  : train {:.8f}, valid: {:.8f}".format(
            item_result['train_loss'], item_result['inter_loss']))
        print("---> DM Loss  : train {:.8f}, valid: {:.8f}".format(
            item_result['train_DMloss'], item_result['inter_DMloss']))
        print("---> LR Loss  : train {:.8f}, valid: {:.8f}".format(
            item_result['train_LRloss'], item_result['inter_LRloss']))
        print("---> DM Cindex: train {:.4f}, valid: {:.4f}".format(
            item_result['train_DMCindex'], item_result['inter_DMCindex']))
        print("---> LR Cindex: train {:.4f}, valid: {:.4f}".format(
            item_result['train_LRCindex'], item_result['inter_LRCindex']))

        epoch_time = epoch_time_end - epoch_time_start
        print('         time in {:.0f}m {:.0f}s'.format(epoch_time // 60, epoch_time % 60))
        savedata = pd.DataFrame(save_alldata)
        with pd.ExcelWriter(save_log_path) as writer:
            savedata.to_excel(writer, sheet_name='traininglog')

        # check if early stop
        if item_result['inter_loss'] < early_valid:
            early_valid = item_result['inter_loss']
            early_epoch = 0
        else:
            early_epoch += 1

        if early_epoch > 5:
            early_stop = True

        torch.cuda.empty_cache()

    end_time = time.time()
    train_time = end_time - start_time

    print('Training complete in {:.0f}m {:.0f}s'.format(train_time // 60, train_time % 60))
