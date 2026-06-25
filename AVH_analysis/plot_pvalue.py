import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

def generate_annulus_points(n, r_min=1.0, r_max=2.0, center=(0, 0)):
    """
    生成环形区域均匀分布点
    :param n: 点数
    :param r_min: 内半径
    :param r_max: 外半径
    :param center: 环形中心坐标 (x0, y0)
    :return: (x, y)坐标数组, r 为半径
    """
    # 生成角度均匀分布
    # np.random.seed(123)
    # theta = np.random.uniform(0, 2 * np.pi, n)
    theta = np.linspace(0, 2 * np.pi, n)

    # 生成半径分布（考虑环形面积因素）
    # 关键公式：r = sqrt( (r_max² - r_min²)*u + r_min² )
    u = np.random.uniform(0, 1, n)
    r = np.sqrt((r_max ** 2 - r_min ** 2) * u + r_min ** 2)

    # 转换为笛卡尔坐标
    x = r * np.cos(theta) + center[0]
    y = r * np.sin(theta) + center[1]

    return x, y, r

def gradient_density(n, r_min, r_max, density_func, center=(0, 0)):
    """
    自定义密度函数分布
    :param density_func: 密度函数 f(r) ∈ [0,1]
    """
    # 生成基础分布
    #  np.random.seed(123)
    theta = np.random.uniform(0, 2 * np.pi, n)
    u = np.random.uniform(0, 1, n)

    # 应用密度函数
    adjusted_u = density_func(u)  # 需确保积分归一化
    r = np.sqrt((r_max ** 2 - r_min ** 2) * adjusted_u + r_min ** 2)
    x = r * np.cos(theta) + center[0]
    y = r * np.sin(theta) + center[1]

    return x,y,r

def rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(*rgb)

def get_pointxy(plotpoint, bw=4, center=(0, 0)):
    index = np.argsort(plotpoint)
    split_r = [0, bw / 2 * 0.25, bw*0.9 / 2 * 0.5, bw*0.9 / 2 * 0.75, bw*0.9 / 2]
    split_num = [len(plotdata) / 4 - 2, len(plotdata) / 4 - 1, len(plotdata) / 4 + 1, len(plotdata) / 4 + 2]
    xp = np.zeros(len(plotpoint))
    yp = np.zeros(len(plotpoint))

    star_i = 0
    for i in range(4):
        n = int(split_num[i])
        rmin = split_r[i]
        rmax = split_r[i+1]
        # xn, yn, _ = generate_annulus_points(n, r_min=rmin, r_max=rmax, center=center)
        xn, yn, _ = gradient_density(n, rmin, rmax, lambda u: 1 - np.exp(-3*u), center=center)
        xp[index[star_i:star_i + n]] = xn
        yp[index[star_i:star_i + n]] = yn
        star_i += n

    return xp, yp

def get_pointxy_new(plotpoint, bw=4.0, rmin=0.2, center=(0, 0)):
    '''
    主要思路是角度均匀地生成多个点，根据点到center的距离，与p值进行匹配，P越小的分配到距离越小的点
    '''
    n = len(plotpoint)
    theta = np.linspace(0, 2 * np.pi, n)
    u = np.random.uniform(0, 1, n)

    r = np.sqrt((bw ** 2 - rmin ** 2) * u + rmin ** 2)
    # 转换为笛卡尔坐标
    x = r * np.cos(theta) + center[0]
    y = r * np.sin(theta) + center[1]
    rsorted = np.argsort(r)
    psorted = np.argsort(plotpoint)

    xp = np.zeros(len(plotpoint))
    xp[psorted] = x[rsorted]
    yp = np.zeros(len(plotpoint))
    yp[psorted] = y[rsorted]

    return xp, yp

basefile = pd.read_excel('../Deeplearning/experiments/Dual_task_CD/cam_mu_test_DM.xlsx')
output = '../Deeplearning/experiments/Dual_task_CD/fig'

critical_values = np.array([0.01, 0.05, 0.1, 0.2])
colors = [rgb_to_hex((8, 51, 110)), rgb_to_hex((16, 92, 164)),
          rgb_to_hex((170, 207, 229)), rgb_to_hex((244, 249, 254))]
new_colors = [rgb_to_hex((170, 207, 229)), rgb_to_hex((201,78,101))]
positions = (critical_values - critical_values[0]) / (critical_values[-1] - critical_values[0])
cmap = mcolors.LinearSegmentedColormap.from_list('custom', list(zip(positions, colors)))

id = basefile['name'].to_list()
split_id = [id[i].split('_') for i in range(len(id))]
split_id = np.reshape(np.concatenate(split_id, axis=0), (-1,3))

ROI_list = ['G0', 'G3', 'G03']
dose_list = ['D40', 'D45', 'D50', 'D55', 'D60', 'D65', 'D70']
ROI_tik = ['G0', 'G3', 'R3']
dose_tik = ['40 Gy', '45 Gy', '50 Gy', '55 Gy', '60 Gy', '65 Gy', '70 Gy']
roinum = len(ROI_list)
dosenum = len(dose_list)

boxwidth = 20
boxspacing = 2
markersize = 50
origin_xlist = [boxwidth / 2 + (boxspacing + boxwidth) * i for i in range(dosenum)]
origin_ylist = [boxwidth / 2 + (boxspacing + boxwidth) * i for i in range(roinum)]

fig, ax = plt.subplots(figsize=(15, 5))
ax.set_facecolor('#F5F5F5')
for i in range(roinum-1):
    plt.axhline(origin_ylist[i]+(boxwidth + boxspacing)/2, c='w', linewidth=2.0)
for j in range(dosenum-1):
    plt.axvline(origin_xlist[j]+(boxwidth + boxspacing)/2, c='w', linewidth=2.0)

for i in range(roinum):
    for j in range(dosenum):
        plotdata = basefile['pvalueTest'].values[1:]

        xall, yall = get_pointxy_new(plotdata, bw=(boxwidth/2 * 0.9), center=(origin_xlist[j], origin_ylist[i]))

        # 分段显示
        nindex = (basefile['pvalueTest'].values[1:] < 0.05) & (basefile['pvalue'].values[1:] < 0.05)
        p1 = plt.scatter(xall[~nindex], yall[~nindex], c=new_colors[0], vmin=0.01, vmax=0.2, # colors[2]
                    s=markersize, marker='o', edgecolors='#F5F5F5', linewidths=0.1)
        p2 = plt.scatter(xall[nindex], yall[nindex], c=new_colors[1], vmin=0.01, vmax=0.2, # colors[1]
                    s=markersize+40, marker='*', edgecolors='#F5F5F5', linewidths=0.1)

ax.legend([p1, p2], ['P > 0.05', 'P < 0.05'], bbox_to_anchor=(0.4, 1), ncol=2)
ax.set_xlim(0, boxwidth * dosenum + (dosenum - 1) * boxspacing)
ax.set_ylim(0, boxwidth * roinum + (roinum - 1) * boxspacing)
ax.set_xticks(origin_xlist)
ax.set_xticklabels(dose_tik, fontsize=15)
ax.set_yticks(origin_ylist)
ax.set_yticklabels(ROI_tik, fontsize=15)
plt.savefig(os.path.join(output, 'DM.png'), dpi=300)

