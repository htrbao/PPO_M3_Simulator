import os
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

train_time = 1
data_path = f'./_saved_stat/hit_mask/new_cal_10layers_32channels_2e-05_4096_not_share_20240926/{train_time}.npy'

hit_map = np.load(data_path)
hit_map[hit_map >= 999] = -1

for i in range(hit_map.shape[0]):
    plt.clf()
    with sns.axes_style("white"):
        ax = sns.heatmap(hit_map[i], annot=True)
    if not os.path.isdir(data_path[:-3]):
        os.makedirs(data_path[:-3])
    plt.savefig(f'{data_path[:-3]}/{i}.png')