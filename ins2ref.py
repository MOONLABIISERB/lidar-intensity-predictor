#import open3d
import numpy as np

#import numpy as np
#from plyreader import PlyReader
#import matplotlib.pyplot as plt
import os
from multiprocessing.dummy import Pool
import sys
sys.path.append('/home/usl/Desktop/random')

import alpha_predictor.alpha_model as alpha_model
import torch
import torch.backends.cudnn as cudnn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmak = False
cudnn.deterministic = True
cudnn.enabled = True
torch.cuda.set_device(0)

#root = '/media/moonlab/sd_card/Rellis_3D_lidar_example/'
models = alpha_model.alpha()
models.to(device)
#print(models)
models.load_state_dict(torch.load('/home/usl/Desktop/random/alpha_predictor/models/best_model_mega_tanh.pth')['model_state_dict'])

def load_bin(bin_path):
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    #print(obj.shape)
    return obj

def alpha_predictor(input):
    data = torch.Tensor(input)
    models.eval()
    inp = data.to(device)
    out = models(inp)
    angles = out.detach().cpu().numpy()
    angles = np.array(angles).flatten()
    return angles

def ref_conv(points):
    #pr = PlyReader()
    #plydata = pr.open(ply_path)
    #vertex =plydata['vertex']
    #print('lol')
    #points = load_bin('/home/usl/Desktop/intensity_salsa/Rellis_3D_os1_cloud_node_kitti_bin/Rellis-3D/00003/os1_cloud_node_kitti_bin/'+bin_path)
    #labels = np.fromfile('/home/usl/Desktop/intensity_salsa/Rellis_3D_os1_cloud_node_semantickitti_label_id_20210614/Rellis-3D/00003/os1_cloud_node_semantickitti_label_id/'+bin_path[:-3]+'label',dtype = np.int32,count = -1)
    #pc = open3d.geometry.PointCloud()
    x,y,z,ins= points[:,0],points[:,1],points[:,2],(points[:,3]*65535)
    rang = np.sqrt(x**2+y**2+z**2)
    dis = x**2 + y**2 + z**2

    #removing lidar points less than 1 meter range(near range effect) 
    ind = np.where(rang < 1)[0]
    ins = np.delete(ins,list(ind))
    x = np.delete(x,list(ind))
    y = np.delete(y,list(ind))
    z = np.delete(z,list(ind))
    rang = np.delete(rang,list(ind))
    dis = np.delete(dis,list(ind))
    #labels = np.delete(labels,list(ind))


    points_n = np.zeros((len(x),3))
    points_n[:,0],points_n[:,1],points_n[:,2] = x , y, z
    ld_vec = points_n/np.sqrt(dis[:,np.newaxis])
    #pc.points = open3d.utility.Vector3dVector(points)
    #pc.estimate_normals(search_param = open3d.geometry.KDTreeSearchParamHybrid(radius = 0.5,max_nn = 10))
    #open3d.visualization.draw_geometries([pc])
    #nm = np.asarray(pc.normals)
    angles = alpha_predictor(ld_vec)
    cal_intensity = ins*dis/np.cos(angles)/6553500

    points_n = np.zeros((len(x),4),dtype = np.float32)
    points_n[:,0], points_n[:,1], points_n[:,2], points_n[:,3] = x,y,z,cal_intensity
    return points_n


