#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 19:18:39 2023

@author: moonlab
"""
import numpy as np

root = '/media/moonlab/sd_card/Rellis_3D_lidar_example/'
data = 'val'
train_data = np.load(root+data+'_data_dotv2_test.npy',allow_pickle = True)
train_gt = np.load(root+data+'_GT_dotv2_test.npy',allow_pickle = True)

train_data = np.delete(train_data,np.where(train_gt <0),0)
train_gt = np.delete(train_gt,np.where(train_gt < 0))

train_data = np.delete(train_data,np.where(train_gt >1),0)
train_gt = np.delete(train_gt,np.where(train_gt > 1))

print(train_gt.mean(),np.median(train_gt),len(train_gt))
np.save(root+data+'_data_dotv2_test.npy',train_data,allow_pickle=True)
np.save(root+data+'_GT_dotv2_test.npy',train_gt,allow_pickle=True)