'''
主要工作：将原始的数据结构整理为：[师兄做了，就没写这部分]
一级文件夹：HN-CHUM-001
二级文件夹：CT, RD, RP, RS
'''
import os
from os.path import join

cach_dir = r'F:\cjj\work\远处转移预测\data\original'
save_dir = r'D:\project\0_Competition\data\DCM_data'

cach_list = os.listdir(cach_dir)

for cach in cach_list:
    ori_dir = join(cach_dir, cach)

    for name in os.listdir(ori_dir):
        save_folder = join(save_dir, name)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        dir_list = list(filter(lambda name: 'TomoTherapy' in name, os.listdir(join(ori_dir, name))))
        dcm_folder = join(ori_dir, name, dir_list[0])

        img_folder = list(filter(lambda name: 'Image' in name, os.listdir(dcm_folder)))
        struct_folder = list(filter(lambda name: 'Structure' in name, os.listdir(dcm_folder)))
        plan_folder = list(filter(lambda name: 'Plan-' in name, os.listdir(dcm_folder)))
        dose_folder = list(filter(lambda name: 'Dose-' in name, os.listdir(dcm_folder)))

        print('{} image: {}, structure: {}, dose: {}, plan: {}'.format(name, len(img_folder),
                                                                       len(struct_folder),
                                                                       len(dose_folder),
                                                                       len(plan_folder)))