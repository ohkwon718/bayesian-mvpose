import json  
import numpy as np 
import torch
from heatmap_subtraction.conv_deconv import conv_deconv, conv_deconv_shortcut, resized_conv_deconv


class HeatmapSubtractor:
    def __init__(self, fullpath_json, device):
        """
        param fullpath_json : str
        param device : torch.device()
        """
        with open(fullpath_json) as json_file:
            settings = json.load(json_file)            

        self.device = device
        fullpath_model = settings["fullpath_model"]
        
        if settings["shortcut"]:
            self.model = resized_conv_deconv(conv_deconv_shortcut(C = settings["layer"], kernal_size=settings["kernel_size"], stride=settings["stride"]).to(device), size_net=settings["imgsize_net"])
        else:
            self.model = resized_conv_deconv(conv_deconv(C = settings["layer"], kernal_size=settings["kernel_size"], stride=settings["stride"]).to(device), size_net=settings["imgsize_net"])

        state = torch.load(settings["fullpath_model"])
        self.model.load_state_dict(state['model_state_dict'])
        self.model.eval()
        
    def get_newcomer(self, batch_simpl3Ds, idxs_part, hmss, camera_matrixs):
        """
        param batch_simpl3Ds : torch.tensor { n x s x 14 x 3 }
        param idxs_part : [ int, ...]
        param hmss : torch.tensor  { v x J x h x w }
        param camera_matrixs : torch.tensor { v x 3 x 4 }
        return torch.tensor  { v x J x h x w }
        """
        v, j, h, w = hmss.shape
        n,s = batch_simpl3Ds.shape[:2]
        N = n*s
        batch_xyz = batch_simpl3Ds.view(N, 14, 3).transpose(0,1)
        batch_xyz1 = torch.cat( ( batch_xyz, torch.ones(14, N, 1).type(batch_xyz.type()) ), -1)
        batch_xy1 = (camera_matrixs[:, None, None, ...] @ batch_xyz1[None, ..., None])[..., 0]
        batch_xy1 /= batch_xy1[...,-1:]
        batch_xy = torch.round(batch_xy1[...,:-1]).short()
        
        x1 = torch.from_numpy(hmss).float().to(self.device)
        x2 = torch.zeros(hmss.shape).to(self.device)
        bool_in = ((batch_xy >= 0).prod(-1) * (batch_xy < torch.tensor([w,h]).to(self.device).type(batch_xy.type()) ).prod(-1)).byte()
        
        idx_v = torch.arange(v)[:,None,None].expand((v,14,N))[bool_in].long()
        idx_j = torch.tensor(idxs_part)[None,:,None].expand((v,14,N))[bool_in].long()
        idx_h = batch_xy[bool_in][...,1].long()
        idx_w = batch_xy[bool_in][...,0].long()
        x2[idx_v, idx_j, idx_h, idx_w] = 1.0
        
        x = torch.cat((x1.view(v*j,1,h,w), x2.view(v*j,1,h,w)), 1)
        res = self.model(x).view(v,j,h,w)
        
        return res