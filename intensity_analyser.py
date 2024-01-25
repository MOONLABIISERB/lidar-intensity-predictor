## Append path that contains the alpha_model.py path
import sys
sys.path.append('./utils')
sys.path.append('./alpha_predictor')
from json import load
import numpy as np
from plyreader import PlyReader
import matplotlib.pyplot as plt
import os
import open3d
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'qt')
import math

import alpha_model
import torch
import torch.backends.cudnn as cudnn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmak = False
cudnn.deterministic = True
cudnn.enabled = True
torch.cuda.set_device(0)

root = '/path/to/lidar-intensity-predictor'
models = alpha_model.alpha()
models.to(device)
models.load_state_dict(torch.load(root+'/alpha_predictor/model/best_model_mega_tanh.pth')['model_state_dict'])

def load_from_bin(bin_path):
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return obj

def alpha_prdictor(normals):
    data = torch.Tensor(normals)
    models.eval()
    angles = []
    inp = data.to(device)
    out = models(inp)
    #print(out)
    angles = out.detach().cpu().numpy()
    #print(angles)
    angles = np.array(angles)
    return angles

def convert_ply2bin(ply_path,bin_path=None):
    pr = PlyReader()
    plydata = pr.open(ply_path)
    vertex =plydata['vertex']
    #print(vertex['intensity'])
    #return vertex['label'],vertex['intensity']
    #plt.scatter(label,i)
    #plt.show()
    #print(type(vertex['label']),type(vertex['intensity']))
    #print('h')
    pc = open3d.geometry.PointCloud()
    x,y,z,ins= np.array(vertex['x']),np.array(vertex['y']),np.array(vertex['z']),np.array(vertex['intensity'])
    points = np.zeros((len(x),3))
    points[:,0],points[:,1],points[:,2] = x , y, z
    pc.points = open3d.utility.Vector3dVector(points)
    pc.estimate_normals(search_param = open3d.geometry.KDTreeSearchParamHybrid(radius = 0.3,max_nn = 10))
    #open3d.visualization.draw_geometries([pc])
    nm = np.asarray(pc.normals)
    nm[:,2] = np.abs(nm[:,2]) ## converting the rogue normals eg: downward normal vectors for grass

    angles = alpha_predictor(nm)

    # try:
    #     normal_z = [math.acos(i) for i in nm[:,2]]
    # except ValueError:
    #     print('error')
    #     labe,f,dist,ins , ia  = 'False', 'False', 'False', 'False', 'False'
    #     return labe,ins, dist, ia
    # else:   
    #print(max(ins))

    dis = np.square(x)+np.square(y)+np.square(z) ##Range Square
    dist = np.abs(np.sqrt(dis)) ## Range of the liDAr points

    # ind = np.where(dis == 0)[0]
    # ins = np.delete(ins,list(ind))
    # dis = np.delete(dis,list(ind))
    # dist = np.delete(dist,list(ind))
    # angles = np.delete(angles,list(ind))
    # nm = np.delete(nm,list(ind),axis = 0)
    # labe = np.delete(vertex['label'],list(ind))
    
    ## Removing the LiDAr points less than 6m range (near range effect)
    ind = np.where(dis <= 36)[0]
    ins = np.delete(ins,list(ind))
    dis = np.delete(dis,list(ind))
    dist = np.delete(dist,list(ind))
    points = np.delete(points,list(ind),axis = 0)
    nm = np.delete(nm,list(ind),axis = 0)
    labe = np.delete(vertex['label'],list(ind))


    ld_vec = points/dist[:,np.newaxis] ##normalising the  pointer vector
    ld_vec = nm - ld_vec  #subtracting the pointer vector from surface normal
    
    
    return labe,ins, dist, ld_vec #dist = distance , dis = distance square, ld_vec = normals - pointer vector
    
if __name__ == "__main__":
    intens = []
    labels = []
    distance = []
    intensity = []
    ang = []
    x = 0
    for i in os.listdir('/path/to/Rellis_3D_os1_cloud_node_color_ply/Rellis-3D/00000/os1_cloud_node_color_ply'):
        if x%3:
            x=x+1
            continue
        label,i, dist, ia = convert_ply2bin('/path/to/Rellis_3D_os1_cloud_node_color_ply/Rellis-3D/00000/os1_cloud_node_color_ply/'+i)
        if label == 'False':
            continue
        else:
            labels.extend(label)
            #intens.extend(i)
            distance.extend(dist)
            intensity.extend(i)
            ang.extend(ia)
        x = x + 1
        print(x)
        if x> 2500:
           break
    print('lol')

    pc_f = []
    di = []
    insen = []
    incident = []
    ant = [3,4,17,18,19,23,27,31,33]
    labs = ['grass','tree','person','fence','bush','concrete','barrier','puddle','mud',]
    col = ['lightgreen','purple','red','orange','green','grey','violet','cyan','salmon']
    for l in ant:
        indices = list(np.where(np.array(labels) == l)[0])
        di.append(np.array([distance[i] for i in indices]))
        insen.append(np.array([intensity[i] for i in indices]))
        incident.append(np.array([ang[i] for i in indices]))
        print(l)
        
        # norm_int = []
        # for i in range(len(di)):
        #     norm_int_a = np.array(insen[i])*np.array(di[i])/np.cos(np.array(incident[i]))/100
        #     norm_int.append(norm_int_a)
    
## Code to calculate the polyfit for max intensity vs range  
    # fits = []
    # for i in range(len(di)):
    #     dit = []
    #     st = []
    #     i = 0
    #     for lm in np.arange(0,1200,1):
    #         cos = np.where(np.round(di[i],1) == lm)
    #         #print(cos[0])
    #         if len(cos[0])>0:
    #             ts = [insen[i][gs] for gs in cos[0]]
    #             st.append(np.max(ts))
    #             dit.append(lm)
    #     if len(dit)>0:  
    #         fit = np.polyfit(dit,st,20)
    #         fits.append(fit)
    #     else:
    #         fits.append([])
    #     break
    #np.save('./fits.npy',np.array(fits,dtype = object),allow_pickle = True)
    
##Extract training and validation data for angle of incidence 

    #fits = np.load('./fits_final.npy',allow_pickle = True)
    #clas = [0,1,2,4,7]
    #gh = []
    #for i in clas:
    #    p = np.poly1d(fits[i])       
    #    for j in range(len(di[i])):
    #        ind = insen[i][j]
    #        mi = p(di[i][j])
    #        ca = ind/mi
    #        #print(ca)
    #        gh.append(ca)
        
    #gt = incident[0]
    #gt = np.append(gt,incident[1],0)
    #gt = np.append(gt,incident[2],0)
    #gt = np.append(gt,incident[4],0)
    #gt = np.append(gt,incident[7],0)
    #print(gt.shape)
    #print(len(gh))
    #np.save('./train_data_dotv2_test.npy',np.asarray(gt),allow_pickle= True)
    #np.save('./train_GT_dotv2_test.npy',np.asarray(gh),allow_pickle = True)


##Plot alpha vs intensity for different distances
    
    #fits = np.load('./fits_final.npy',allow_pickle = True)
    #clas = 0
    #p = np.poly1d(fits[0])
    # disn = [7,10,15,20,30,40]
    # gh = []
    # ch = []
    #for j in range(len(di[0])):
    #    ind = insen[0][j]
    #    mi = p(di[0][j])
    #    ca = ind/mi
        #cg = incident[0][j]
        #clas += abs(ca-cg)
        #print(ca-cg)
    #print(clas/len(di[0]))
    #     ca = np.arccos(ca)
    #     ca = ca*360/(2*np.pi)
    #     gh.append(oi)
    #     ch.append(ca)
    #     plt.scatter(ca,oi/max(oi),label = str(j)+' m')
        
    # #plt.scatter(ch[0],gh[0])
    # plt.xlabel('Angle of Incidence(degree)')
    # plt.ylabel('Intensity')
    # plt.title('Intensity vs Angle of Incidence')
    # plt.legend()
    # plt.show()
    
## Plot Intensity vs angle of incidence predictor

    # fits = np.load('./fits_final.npy',allow_pickle = True)
    # disn = 10
    # gh = []
    # ch = []
    # for j in range(len(di)):
    #     p = np.poly1d(fits[j])
    #     ind = np.where(np.round(di[j]) == disn)[0]
    #     oi = np.array([insen[j][b] for b in ind])
    #     mi = p(disn)
    #     ca = oi/mi
    #     ca = np.arccos(ca)
    #     ca = ca*360/(2*np.pi)
    #     #gh.append(oi)
    #     #ch.append(ca)
    #     plt.scatter(ca,oi,label = labs[j])
        
    # #plt.scatter(ch[0],gh[0])
    # plt.xlabel('Angle of Incidence(degree)')
    # plt.ylabel('Intensity')
    # plt.title('Intensity vs Angle of Incidence')
    # plt.legend()
    # plt.show()
        
     
## Extract calibrated intensity ranges for different class predictor.

    # norm_int = []
    # for m in range(len(fits)):
    #     p = np.poly1d(fits[m])
    #     cls_int = []
    #     for jp in np.arange(6,40,0.25):
    #         val = p(jp)
    #         #print(val)
    #         cls_int.append(int(val*(jp**2))/100)
    #     norm_int.append(cls_int)
    # k = 0
    # ks = []
    # for pc in norm_int:
    #     #print(pc)
    #     un, inst = np.unique(np.array(pc),return_counts = True)
    #     inst = np.delete(inst,np.where(un <=0))
    #     un = np.delete(un,np.where(un <=0))
    #     #un = np.delete(un,np.where(inst ==2))
    #     #inst = np.delete(inst,np.where(inst ==2))
    #     mean = np.mean(un)
    #     median = np.median(un)
    #     sd = np.std(un)
    #     ks.extend(un)
    #     print((mean,sd,median,labs[k]))
    #     inst = np.delete(inst,np.where(un < mean - 0.8*sd))
    #     un = np.delete(un,np.where(un < mean - 0.8*sd))
    #     inst = np.delete(inst,np.where(un > mean + 0.8*sd))
    #     un = np.delete(un,np.where(un > mean + 0.8*sd))

        # print(len(un),len(inst))
        # if len(inst):
        #     #plt.scatter(inst,un,alpha = 0.8,label = labs[k],color = col[k])
        #     plt.bar(un,inst/np.max(inst),alpha = 1,width = 1,label = labs[k],color = col[k])
        #     plt.legend()
        #     plt.xlim(0, 1300)
        #     plt.xlabel('Calibrated Intensity')
        #     plt.ylabel('Normalized occurences')
        #     plt.savefig('./results1/'+labs[k]+'_5000_1.png')
        #     plt.clf()
        #k = k+1
    #print((np.mean(ks),np.std(ks)))
    # #plt.show()
    
