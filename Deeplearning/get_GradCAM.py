import torch
import torch.nn.functional as F
import torchio as tio
import numpy as np
import SimpleITK as sitk
import os
from Deeplearning.models.dual_task_model import generate_parallel_model
from Deeplearning.dataloader.read_data import read_follow_data, read_data, get_transform

class GradCAMHook3D():
    def __init__(self, module):
        self.features = None
        self.gradients = None
        self.forward_hook = module.register_forward_hook(self.save_features)
        self.backward_hook = module.register_full_backward_hook(self.save_gradients)

    def save_features(self, module, input, output):
        self.features = output.detach()

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

def generate_gradcam_3d(features, gradients, orisize=(64, 128, 128)):
    weights = torch.mean(gradients, dim=(2, 3, 4), keepdim=True)
    cam = torch.sum(weights * features, dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(
        cam,
        size=orisize,
        mode="trilinear",
        align_corners=False,
    )
    cam = cam - torch.min(cam)
    cam = cam / (torch.max(cam) + 1e-8)

    return cam.squeeze().cpu().numpy()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定使用哪一块gpu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = r'E:\LHL\competition\CJJ\Data\CT_RD_nii_resample222'
    data_outcome_dir = r'E:\LHL\competition\CJJ\Data\ALL_dm_lr.xlsx'
    save_dir = os.path.join('./experiments', 'Dual_task_CD')
    model_name = 'inter_best_cindex_model'
    path_checkpoint = os.path.join(save_dir, '{}.pth'.format(model_name))

    save_base_dir = os.path.join(save_dir, 'Grad_CAM')
    if not os.path.exists(save_base_dir):
        os.mkdir(save_base_dir)

    HGJ_name, HGJ_event_DM, HGJ_time_DM, HGJ_event_LR, HGJ_time_LR = read_follow_data(data_outcome_dir, 'HGJ')
    CHUS_name, CHUS_event_DM, CHUS_time_DM, CHUS_event_LR, CHUS_time_LR = read_follow_data(data_outcome_dir, 'CHUS')
    HMR_name, HMR_event_DM, HMR_time_DM, HMR_event_LR, HMR_time_LR = read_follow_data(data_outcome_dir, 'HMR')
    CHUM_name, CHUM_event_DM, CHUM_time_DM, CHUM_event_LR, CHUM_time_LR = read_follow_data(data_outcome_dir, 'CHUM')

    test_subject = read_data(
        data_dir, np.concatenate((HGJ_name, CHUS_name, HMR_name, CHUM_name)),
        np.concatenate((HGJ_event_DM, CHUS_event_DM, HMR_event_DM, CHUM_event_DM)),
        np.concatenate((HGJ_time_DM, CHUS_time_DM, HMR_time_DM, CHUM_time_DM)),
        np.concatenate((HGJ_event_LR, CHUS_event_LR, HMR_event_LR, CHUM_event_LR)),
        np.concatenate((HGJ_time_LR, CHUS_time_LR, HMR_time_LR, CHUM_time_LR)))
    test_dataset = tio.SubjectsDataset(test_subject, transform=get_transform())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                              shuffle=False, num_workers=0, pin_memory=True)

    model = generate_parallel_model(n_input_channels=2, num_classes=1)
    checkpoint = torch.load(path_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['net'])
    model = model.to(device)
    model.eval()

    hook_dm = GradCAMHook3D(model.layer3_DM)
    hook_lr = GradCAMHook3D(model.layer3_LR)
    for i, subjects_batch in enumerate(test_loader):
        name = subjects_batch['name'][0]
        save_dir = os.path.join(save_base_dir, name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # load data
        cts = subjects_batch['ct'][tio.DATA]
        cts = cts.permute(0, 1, 4, 2, 3)
        masks = subjects_batch['mask'][tio.DATA]
        masks = masks.permute(0, 1, 4, 2, 3)
        doses = subjects_batch['dose'][tio.DATA]
        doses = doses.permute(0, 1, 4, 2, 3)
        inputs1 = torch.cat((cts, masks), dim=1)
        inputs1 = inputs1.to(device, non_blocking=True)
        inputs2 = torch.cat((doses, masks), dim=1)
        inputs2 = inputs2.to(device, non_blocking=True)
        inputs = [inputs1, inputs2]

        outDM, outLR = model(inputs)

        model.zero_grad()
        outDM.backward(retain_graph=True)
        cam_dm = generate_gradcam_3d(hook_dm.features, hook_dm.gradients, orisize=(64, 128, 128))

        model.zero_grad()
        outLR.backward()
        cam_lr = generate_gradcam_3d(hook_lr.features, hook_lr.gradients, orisize=(64, 128, 128))

        cam_dm_img = sitk.GetImageFromArray(cam_dm)
        sitk.WriteImage(cam_dm_img, os.path.join(save_dir, 'CAM_Layer3DM.nii.gz'))
        cam_lr_img = sitk.GetImageFromArray(cam_lr)
        sitk.WriteImage(cam_lr_img, os.path.join(save_dir, 'CAM_Layer3LR.nii.gz'))

        img = cts.squeeze().cpu().detach().numpy()
        img_img = sitk.GetImageFromArray(img)
        sitk.WriteImage(img_img, os.path.join(save_dir, 'CT.nii.gz'))

        dose = doses.squeeze().cpu().detach().numpy()
        dose_img = sitk.GetImageFromArray(dose)
        sitk.WriteImage(dose_img, os.path.join(save_dir, 'Dose.nii.gz'))

        mask = masks.squeeze().cpu().detach().numpy()
        mask_img = sitk.GetImageFromArray(mask)
        sitk.WriteImage(mask_img, os.path.join(save_dir, 'Mask.nii.gz'))

    hook_dm.remove()
    hook_lr.remove()
