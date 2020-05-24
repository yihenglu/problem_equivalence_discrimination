#!/usr/bin/env python 
# -*- coding:utf-8 -*-
'''
绘制训练过程
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def draw_train_process():
    total_f1_list = np.load('data/total_f1.npy').tolist()
    # print(total_f1_list)
    brief_f1_list = []
    for i in range(len(total_f1_list)):
        if i % 10 == 0:
            brief_f1_list.append(total_f1_list[i])
    epoch_list = [i for i in range(1, len(brief_f1_list)+1)]
    # total_f1_df = pd.DataFrame({'epoch':epoch_list, 'f1':total_f1_list})
    breif_f1_df = pd.DataFrame({'epoch':epoch_list, 'f1':brief_f1_list})

    sns.set(style="darkgrid")
    # sns.lineplot(epoch_list, total_f1_list)
    sns.lineplot(x="epoch", y="f1", data=breif_f1_df)

    plt.show()
    return

draw_train_process()
