import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

excel_file = pd.read_excel('../Deeplearning/experiments/Dual_task_CD/gradcam_cal_DM.xlsx')
save_dir = '../Deeplearning/experiments/Dual_task_CD/fig'
patient_list = ['HN-HMR-011', 'HN-HMR-014']
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

select_index = list(filter(lambda i: excel_file['ID'][i] in patient_list, range(len(excel_file['ID']))))
basedata = excel_file.iloc[select_index, :9]
plotdata = excel_file.iloc[select_index, 9:]
label_list = list(map(lambda name: name.split("_W")[0], plotdata.columns.values))
label_list = np.unique(np.array(label_list))
Wlist = np.arange(0, 1, 0.01)

ROIG = ['G0', 'G3', 'G03']
plotD = ['D45', 'D50', 'D55', 'D60', 'D65', 'D70']
color_list = ['blue', 'red']
legend_list = ['Low risk', 'High risk']

plot1 = np.array(plotdata.iloc[0, :])
plot1 = np.reshape(plot1, (-1, 100))

plot2 = np.array(plotdata.iloc[1, :])
plot2 = np.reshape(plot2, (-1, 100))
Line_w = 3.0
for i in range(len(plotD)):
    for j in range(len(ROIG)):
        flag = '{}_{}'.format(ROIG[j], plotD[i])
        plot_index = np.where(label_list == flag)[0]

        fig, axs = plt.subplots(1, 1, figsize=(10, 8))
        plot_data1 = plot1[plot_index, :]
        flag1 = basedata.iloc[0, 1]
        plt.plot(Wlist, plot_data1[0], linestyle='-', lw=Line_w,
                 color=color_list[flag1], label=legend_list[flag1])
        plot_data2 = plot2[plot_index, :]
        flag2 = basedata.iloc[1, 1]
        plt.plot(Wlist, plot_data2[0], linestyle='-', lw=Line_w,
                 color=color_list[flag2], label=legend_list[flag2])
        # axs.set_title(flag, fontsize=25)
        axs.set_yticks(np.arange(0, 101, 20))
        axs.set_ylim([0, 100])
        axs.set_xticks(np.arange(0, 1.1, 0.1))
        axs.set_xlim([0, 1])
        axs.tick_params(axis='both', labelsize=20)
        axs.legend(loc='upper right', prop={'size': 20})
        axs.set_xlabel('The value of the attention map', fontsize=25)
        axs.set_ylabel('Cumulative relative volume', fontsize=25)
        fig.tight_layout(h_pad=3)
        plt.savefig(os.path.join(save_dir, '{}.png'.format(flag)), format='png')