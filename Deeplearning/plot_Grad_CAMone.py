import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
import cv2

cam_dir = '../Deeplearning/experiments/Dual_task_CD/Grad_CAM'
save_dir = '../Deeplearning/experiments/Dual_task_CD/fig'

patient = 'HN-HMR-039'
z = 57
xlim = [30, 110]
ylim = [105, 25]

ct = sitk.ReadImage(join(cam_dir, patient, 'CT.nii.gz'))
dose = sitk.ReadImage(join(cam_dir, patient, 'Dose.nii.gz'))
mask = sitk.ReadImage(join(cam_dir, patient, 'Mask.nii.gz'))
cam = sitk.ReadImage(join(cam_dir, patient, 'CAM_layer3.nii.gz'))
cam_arr = sitk.GetArrayFromImage(cam)
ct_arr = sitk.GetArrayFromImage(ct)
dose_arr = sitk.GetArrayFromImage(dose) * 87
mask_arr = sitk.GetArrayFromImage(mask)

ct_plot = ct_arr[z, :, :]
dose_plot = dose_arr[z, :, :]
cam_plot = cam_arr[z, :, :]
mask_plot = mask_arr[z, :, :]


dx, dy = np.where(dose_plot)
dosex, dosey = np.meshgrid(np.unique(dx), np.unique(dy))
dosez = dose_plot[dosex, dosey]

contours, _ = cv2.findContours(mask_plot.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

save_path = join(save_dir, '{}_CT_{}.png'.format(patient, z))
figct, axct = plt.subplots(1, 1)
ax = axct.imshow(ct_plot, cmap='gray', origin='upper', clim=(0, 1))
axct.set_xlim(xlim)
axct.set_ylim(ylim)
axct.set_axis_off()
figct.colorbar(ax)
plt.savefig(save_path, format='png')
plt.close(figct)

save_path = join(save_dir, '{}_Dose_{}.png'.format(patient, z))
figd, axd = plt.subplots(1, 1)
ax = axd.imshow(dose_plot, cmap='jet', origin='upper', clim=(0, 80))
axd.set_xlim(xlim)
axd.set_ylim(ylim)
axd.set_axis_off()
figd.colorbar(ax)
plt.savefig(save_path, format='png')
plt.close(figd)

save_path = join(save_dir, '{}_CAM_{}.png'.format(patient, z))
figd, axd = plt.subplots(1, 1)
ax = axd.imshow(cam_plot, cmap='jet', origin='upper', clim=(0, 1))
axd.set_xlim(xlim)
axd.set_ylim(ylim)
axd.set_axis_off()
figd.colorbar(ax)
plt.savefig(save_path, format='png')
plt.close(figd)

save_path = join(save_dir, '{}_CAMCD_{}.png'.format(patient, z))
figcc, axcc = plt.subplots(1, 1)
axcc.imshow(ct_plot, cmap='gray', origin='upper', alpha=1, clim=(0, 1))
axcc.contour(dosey, dosex, dosez, [60, 70], origin='upper', alpha=1,
             linestyles=['dashed', 'solid'], colors='yellow',
             linewidths=1.5)
ax = axcc.imshow(cam_plot, cmap='jet', origin='upper', alpha=0.5, clim=(0, 1))
for j in range(len(contours)):
    cx = np.append(contours[j][:, 0, 0], contours[j][:, 0, 0][0])
    cy = np.append(contours[j][:, 0, 1], contours[j][:, 0, 1][0])
    axcc.plot(cx, cy, 'r-', linewidth=1.5)
axcc.set_xlim(xlim)
axcc.set_ylim(ylim)
axcc.set_axis_off()
figcc.colorbar(ax)
plt.savefig(save_path, format='png')
plt.close(figcc)