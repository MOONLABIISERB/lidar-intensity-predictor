
import numpy as np
import os
import alpha_predictor.alpha_model as alpha_model
import torch
import torch.backends.cudnn as cudnn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmak = False
cudnn.deterministic = True
cudnn.enabled = True
torch.cuda.set_device(device)

models = alpha_model.alpha()
models.to(device)
models.load_state_dict(torch.load('./alpha_predictor/models/best_model_mega_tanh.pth')['model_state_dict'])

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
    x,y,z,ins= points[:,0],points[:,1],points[:,2],(points[:,3]*65535)
    range = np.sqrt(x**2+y**2+z**2)
    dis = x**2 + y**2 + z**2

    #removing lidar points less than 1 meter range(near range effect) 
    ind = np.where(range < 1)[0]
    ins = np.delete(ins,list(ind))
    x = np.delete(x,list(ind))
    y = np.delete(y,list(ind))
    z = np.delete(z,list(ind))
    range = np.delete(rang,list(ind))
    dis = np.delete(dis,list(ind))


    points_n = np.zeros((len(x),3))
    points_n[:,0],points_n[:,1],points_n[:,2] = x , y, z
    ld_vec = points_n/np.sqrt(dis[:,np.newaxis])
    
    #predicting angle of incidence.
    angles = alpha_predictor(ld_vec)
    #claibrating intensity for reflectivity
    cal_intensity = ins*dis/np.cos(angles)/6553500

    points_n = np.zeros((len(x),4),dtype = np.float32)
    points_n[:,0], points_n[:,1], points_n[:,2], points_n[:,3] = x,y,z,cal_intensity
    return points_n


