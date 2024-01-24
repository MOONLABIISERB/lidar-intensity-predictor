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
from tqdm import tqdm
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'qt')
import sys
sys.path.append('./alpha_predictor/')
import statistics as st

import alpha_model
import torch
import torch.backends.cudnn as cudnn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmak = False
cudnn.deterministic = True
cudnn.enabled = True
torch.cuda.set_device(0)


labs_median = [50,20,1600,300,140]  #hand optimized value
labs = ['bush','puddle','person','grass','tree']

z_factor = [0,1,0,1,0]
height = -0.5

def binarysearch(arr, m, x):
    l, r = 0, m - 1
     
    while l <= r:
        mid = (l + r) // 2
         
        # Checking if the middle element is equal to x
        if arr[mid] == x:
            return True
        elif arr[mid] < x:
            l = mid + 1
        else:
            r = mid - 1
             
   # return true , if element x is present in the array
   # else false
    return False
 
# Function to count common element
def count_common(a, n, b, m):
    b.sort()
    count = 0
     
    # Iterate each element of array a
    for i in range(n):
       
        # Checking  if the element of array a is present in
        # array b using the binary search function
        if binarysearch(b, m, a[i]):
             count += 1
    # Return count of common element
    return count
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

def class_predictor(cal_ins,z): ##predict the class of the LIDAR points comparing the defined medians
    closest_elements = find_closest_element(np.array(labs_median), cal_ins)
    for i in range(len(labs)):
        ind = np.where(closest_elements==labs_median[i])[0]
        heig = z[ind]
        if z_factor[i] == 1:
            inx = np.where(heig>height)[0]
            closest_elements[ind[inx]] = 4
            ind = np.delete(ind,inx)
        else:
            inx = np.where(heig<height)[0]
            closest_elements[ind[inx]] = 3
            ind = np.delete(ind,inx)
        closest_elements[ind] = ant[i]
    return np.array(closest_elements).flatten()

def label_assigner(labels):   #function to give coloured annotation to the lidar points
    gt_label_col = np.zeros((len(labels),3))
    for i in range(len(labs)):
        ind = np.where(labels==ant[i])[0]
        gt_label_col[ind] = colors[i]
    return gt_label_col

def load_from_bin(bin_path): #loading the bin file
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return obj

def neighbour_filter(pc,labels): ##filter outliers in prediction considering neighbours
    pcd = open3d.geometry.KDTreeFlann(pc)
    label_a = np.array([17]) ##input classes to be filtered, 17 is for person.
    for i in range(len(labels)):
        ml = labels[i]
        if ml == label_a.any():
            rad = 0.7
        else:
            rad = 0.25
        [k,idx,_] = pcd.search_radius_vector_3d(pc.points[i],rad)
        if len(idx) == 1:
            continue
        ml = labels[i]
        ls = labels[idx[1:]]
        lk = st.mode(ls)
        if ml == lk:
            continue 
        else:
            labels[i] = lk
    return labels

def convert_ply2bin(bin_path,label_file): ##read, extract and filter the point cloud
    raw_points = load_from_bin(bin_path)
    labels = np.fromfile('/path/to/Rellis_3D_vel_cloud_node_semantickitti_label_id/Rellis-3D/00000/vel_cloud_node_semantickitti_label_id/'+label_file[:-3]+'label',dtype = np.int32,count = -1)
    
    ind = np.where(labels == 0)[0]
    labels = np.delete(labels,list(ind))
    raw_points = np.delete(raw_points,list(ind),axis = 0)
    
    # extracting the points and intensity
    x,y,z,ins= raw_points[:,0],raw_points[:,1],raw_points[:,2],raw_points[:,3]
    dis = np.square(x)+np.square(y)+np.square(z)
    dist = np.abs(np.sqrt(dis))
    
    ##removing lidar points less that 7 meters(near range effect)
    ind = np.where(dis <= 50)[0]
    ins = np.delete(ins,list(ind))
    dis = np.delete(dis,list(ind))
    dist = np.delete(dist,list(ind))
    x = np.delete(x,list(ind))
    y = np.delete(y,list(ind))
    z = np.delete(z,list(ind))
    labels = np.delete(labels,list(ind))
    
    ##removing lidar points greater than 50 meters
    ind = np.where(dis >= 2500)[0]
    ins = np.delete(ins,list(ind))
    dis = np.delete(dis,list(ind))
    dist = np.delete(dist,list(ind))
    x = np.delete(x,list(ind))
    y = np.delete(y,list(ind))
    z = np.delete(z,list(ind))
    labels = np.delete(labels,list(ind))
    
    
    ## removing classes that are not considered.
    intensity = []
    x1 = []
    y1 = []
    z1 = []
    dis2 = []
    label = []
    
    for i in ant:
        ind = np.where(labels == i)
        x1.extend(x[ind])
        y1.extend(y[ind])
        z1.extend(z[ind])
        dis2.extend(dis[ind])
        intensity.extend(ins[ind])
        label.extend(labels[ind])
        
    x1 = np.array(x1)
    y1 = np.array(y1)
    z1 = np.array(z1)
    dis2 = np.array(dis2)
    intensity = np.array(intensity)
    label = np.array(label)
    
    return dis2,x1,y1,z1, label, intensity ##distance^2, x,y,z, alpha, labels, filtered intensity, point cloud

def normal_estimate(x,y,z):
    ## Normal estimation of the point cloud
    points = np.zeros((len(x),3))
    points[:,0],points[:,1],points[:,2] = x , y, z
    pc = open3d.geometry.PointCloud()
    pc.points = open3d.utility.Vector3dVector(points)
    pc.estimate_normals(search_param = open3d.geometry.KDTreeSearchParamHybrid(radius = 0.5,max_nn = 10),fast_normal_computation = False )
    #open3d.visualization.draw_geometries([pc])
    nm = np.asarray(pc.normals) ## Array of normals
    return nm , pc ## normals, point cloud

def predict_alpha(normals):
    ## Angle of incidence predictor using surface normals only!
    data = torch.Tensor(normals)
    models.eval()
    inp = data.to(device)
    out = models(inp)
    angles = out.detach().cpu().numpy()
    angles = np.array(angles).flatten()
    return angles ##angle of incidence

def vel2ouster(intensity, distance,vel2os):
    f_qt = []
    for k in range(len(intensity)):
        intes = intensity[k]
        dic = distance[k]
        qot = vel2os(dic)
        intes = intes*qot
        f_qt.append(intes)
    return np.asarray(f_qt)
    
def Iou(labels,predict):
    class_iou = []
    for i in ant:
        gt_ids = np.where(labels == i)[0]
        if len(gt_ids) <5:
            continue
        pred_ids = np.where(predict == i)[0]
        intersec = count_common(gt_ids, len(gt_ids), pred_ids, len(pred_ids))
        cl_iou = intersec/(len(gt_ids)+len(pred_ids)-intersec)
        class_iou.append(cl_iou)
    return class_iou

if __name__ == "__main__":
    acc = []
    j = 0
    root = '/media/moonlab/sd_card/Rellis_3D_lidar_example/'
    models = alpha_model.alpha()
    models.to(device)
    models.load_state_dict(torch.load(root+'./alpha_predictor/model/best_model_mega_tanh.pth')['model_state_dict'])
    
    ant = [19,31,17,3,4] #class id to be considered
    ants = [3,4,17,18,19,23,27,31,33] ##total number of classes in the dataset
    
    vel2os_fit = np.load('./fit_tree_vel.npy',allow_pickle = True)
    vel2os = np.poly1d(vel2os_fit)
    colors = [[255,255,0],[0,255,0],[255,0,0],[0,255,255],[0,0,255]]
    for i in os.listdir('./Rellis_3D_vel_cloud_node_kitti_bin/Rellis-3D/00000/vel_cloud_node_kitti_bin'):
        dis2, x, y, z, labels, intensity = convert_ply2bin('./Rellis_3D_vel_cloud_node_kitti_bin/Rellis-3D/00000/vel_cloud_node_kitti_bin/'+i,i)
        
        normals, pc = normal_estimate(x,y,z) ##estimate the normals
        angles = predict_alpha(normals) ##predict the angle of incidence
        intensity = vel2ouster(intensity,np.sqrt(dis2),vel2os) ##Convert velodyne intensity to Ouster form
        
        cal_intensity = intensity*dis2/np.cos(angles)/100 ##calibrated intensity; devision by 100 is to scale down the values
        
        pred_labels = class_predictor(cal_intensity,z) ##predict the labels of the point cloud
        pred_labels = neighbour_filter(pc,pred_labels) ##filter the outliers

        label_col = label_assigner(pred_labels) ##colour assigner for visualization
        gt_labels = label_assigner(labels)
        
        ## Visualizer
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
        
        ##Calculate the mIoU
        class_miou = Iou(labels,pred_labels)
        print('Iou of each class:',class_miou)
        frame_miou = sum(class_miou)/len(class_miou)
        print('mIoU of the current frame: ',frame_miou)
        acc.append(frame_miou)
    print(np.mean(acc))
    
        
        
