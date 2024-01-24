#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 14:34:39 2023

@author: moonlab
"""

import numpy as np
from plyreader import PlyReader
import matplotlib.pyplot as plt
import os
import open3d

import sys
sys.path.append('/media/moonlab/sd_card/')

import alpha_model
import torch
import torch.backends.cudnn as cudnn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmak = False
cudnn.deterministic = True
cudnn.enabled = True
torch.cuda.set_device(0)

root = '/media/moonlab/sd_card/Rellis_3D_lidar_example/'
models = alpha_model.alpha()
models.to(device)
models.load_state_dict(torch.load(root+'/model/best_model_mega_tanh.pth')['model_state_dict'])

ant = [19,31,17,4,3]
ants = [3,4,17,18,19,23,27,31,33]

colors = [[255,255,0],[0,255,0],[255,0,0],[0,0,255],[0,255,255]]
labs = ['bush','puddle','person','tree','grass']
height = 2.5
labs_mean = [588,463,422,752,667]
labs_std = [88,77,91,148,115]
#labs_median = [451,688,377,466,696] #median of all the values
labs_median = [160,100,1800,250,1000]
#labs_median = [403,755,399,500,682] #median fro 500 datapoints
z_factor = [1,0,0,1,0]
fits = np.load('./fits_final.npy',allow_pickle = True)

def find_closest_element(arr1, arr2):
    # Reshape arr1 and arr2 to have dimensions for broadcasting
    arr1 = arr1.reshape(-1, 1)
    arr2 = arr2.reshape(1, -1)

    # Compute the absolute differences between elements in arr1 and arr2
    absolute_diff = np.abs(arr1 - arr2)

    # Find the indices of the minimum differences along axis 0
    closest_indices = np.argmin(absolute_diff, axis=0)

    # Get the closest elements from arr1 using the indices
    closest_elements = arr1[closest_indices]

    return closest_elements

def class_predictor(cal_ins,z):
    #for i in range(len(labs)):
        #ind = np.where((np.logical_and(labs_mean[i]-labs_std[i]<=cal_ins,cal_ins<=labs_mean[i]+labs_std[i])))
    closest_elements = find_closest_element(np.array(labs_median), cal_ins)
    label_col = np.zeros((len(cal_ins),3))
    for i in range(len(labs)):
        ind = np.where(closest_elements==labs_median[i])[0]
        closest_elements[ind] = ant[i]
        label_col[ind] = colors[i]
    return np.array(closest_elements).flatten(),label_col

def label_assigner(labels):
    gt_label_col = np.zeros((len(labels),3))
    for i in range(len(labs)):
        ind = np.where(labels==ant[i])[0]
        gt_label_col[ind] = colors[i]
    return gt_label_col

def load_from_bin(bin_path):
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return obj

def alpha_predictor(input):
    data = torch.Tensor(input)
    models.eval()
    inp = data.to(device)
    out = models(inp)
    angles = out.detach().cpu().numpy()
    angles = np.array(angles).flatten()
    return angles

def convert_ply2bin(ply_path,bin_path=None):
    pr = PlyReader()
    plydata = pr.open(ply_path)
    vertex =plydata['vertex']
    pc = open3d.geometry.PointCloud()
    x, y, z, ins, labe = np.array(vertex['x']),np.array(vertex['y']),np.array(vertex['z']),np.array(vertex['intensity']),  np.array(vertex['label'])
    dis = np.square(x)+np.square(y)+np.square(z)
    dist = np.abs(np.sqrt(dis))
    
    ind = np.where(dis <= 50)[0]
    ins = np.delete(ins,list(ind))
    dis = np.delete(dis,list(ind))
    dist = np.delete(dist,list(ind))
    x = np.delete(x,list(ind))
    y = np.delete(y,list(ind))
    z = np.delete(z,list(ind))
    labe = np.delete(labe,list(ind))
    
    # ind = np.where(dis >= 1600)[0]
    # ins = np.delete(ins,list(ind))
    # dis = np.delete(dis,list(ind))
    # dist = np.delete(dist,list(ind))
    # x = np.delete(x,list(ind))
    # y = np.delete(y,list(ind))
    # z = np.delete(z,list(ind))
    # labe = np.delete(labe,list(ind))
    
    intensity = []
    x1 = []
    y1 = []
    z1 = []
    dis2 = []
    label = []
    
    for i in ant:
        ind = np.where(labe == i)
        x1.extend(x[ind])
        y1.extend(y[ind])
        z1.extend(z[ind])
        dis2.extend(dis[ind])
        intensity.extend(ins[ind])
        label.extend(labe[ind])
        
    x1 = np.array(x1)
    y1 = np.array(y1)
    z1 = np.array(z1)
    dis2 = np.array(dis2)
    intensity = np.array(intensity)
    label = np.array(label)
    

    points = np.zeros((len(x1),3))
    points[:,0],points[:,1],points[:,2] = x1 , y1, z1
    ld_vec = points/np.sqrt(dis2[:,np.newaxis])
    #pc.points = open3d.utility.Vector3dVector(points)
    #pc.estimate_normals(search_param = open3d.geometry.KDTreeSearchParamHybrid(radius = 0.5,max_nn = 10))
    #open3d.visualization.draw_geometries([pc])
    #nm = np.asarray(pc.normals)
    angles = alpha_predictor(ld_vec)
    return dis2,x1,y1,z1,angles, label, intensity,pc

if __name__ == "__main__":
    acc = []
    j = 0
    for i in os.listdir('./Rellis_3D_os1_cloud_node_color_ply/Rellis-3D/00000/os1_cloud_node_color_ply'):
        #print(i)
        dis2, x, y, z, angles, labels, intensity,pc = convert_ply2bin('./Rellis_3D_os1_cloud_node_color_ply/Rellis-3D/00000/os1_cloud_node_color_ply/'+i)
        cal_intensity = intensity*dis2/np.cos(angles)/100
        pred_labels,label_col = class_predictor(cal_intensity,z)
        gt_labels = label_assigner(labels)
        pc.colors = open3d.utility.Vector3dVector(gt_labels/255.)
        open3d.visualization.draw_geometries([pc])
        pc.colors = open3d.utility.Vector3dVector(label_col/255.)
        open3d.visualization.draw_geometries([pc])

        
        # fig, ax = plt.subplots(figsize=(24,24))
        # #ax = fig.add_subplot(projection='2d')
        # ax.scatter(x, y, c=gt_labels/255.)
        # plt.xlim(-40, 40)
        # plt.ylim(-40, 40)

        # #plt.axis('off')
        
        # try:
        #     fig.savefig('./intensity_pred/'+str(j)+'_g.png')
        # except ValueError:
        #     plt.close(fig)
        # else:
        #     plt.close(fig)

        # fig, ax = plt.subplots(figsize=(10,10))
        # #ax = fig.add_subplot(projection='2d')
        # ax.scatter(x, y, c=label_col/255.)
        # #plt.axis('off')
        # fig.savefig('./intensity_pred/'+str(j)+'_p.png')
        # plt.close(fig)

        res = labels==pred_labels
        acc.append(sum(bool(x) for x in res)/len(res)*100)
        print('Accuracy: ',sum(bool(x) for x in res)/len(res)*100)
        j = j+1
        if j%500 == 0:
            break
    print(np.mean(acc))
    
        
        
