import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_pose_prediction(device):
    pose_prediction = pose_prediction_rnn(backbone_GRU(42, 2048, 1, [42])).to(device)
    file_model = "pose_prediction/model_prediction_rnn.pt"
    state = torch.load(file_model)
    pose_prediction.load_state_dict(state['model_state_dict'])
    pose_prediction.eval()
    return pose_prediction

class backbone_GRU(nn.Module):
    def __init__(self, input_size=42, gru_hidden_size=168, gru_num_layers=1, fc_layers=[168, 42]):
        super(backbone_GRU,self).__init__()
        self.gru = nn.GRU(input_size, gru_hidden_size, gru_num_layers)
        self.add_module("gru", self.gru)
        
        fc_layers = fc_layers.copy()
        fc_layers.insert(0, gru_hidden_size)
        self.fcs = []
        self.bns = []
        for i in range(1, len(fc_layers)-1):
            self.fcs.append(nn.Linear(fc_layers[i-1], fc_layers[i]))
            self.add_module("fc{:02d}"+str(i), self.fcs[-1])
            self.bns.append(nn.BatchNorm1d(fc_layers[i]))
            self.add_module("bn{:02d}"+str(i), self.bns[-1])
        self.fcs.append(nn.Linear(fc_layers[-2], fc_layers[-1]))
        self.add_module("fc{:02d}"+str(len(fc_layers)), self.fcs[-1])
            
    def forward(self, x, h = None, p=0.0, dropout = True):
        """
        param x : (seq_len, batch, gru_input_size)
        """
        x, h = self.gru(x, h)
        x = x.squeeze(0)
        x = F.dropout(x, p=p, training = dropout)
        for i in range(len(self.fcs)-1):
            fc = self.fcs[i]
            bn = self.bns[i]
            x = F.relu(bn(fc(x)))
            x = F.dropout(x, p=p, training = dropout)
        x = self.fcs[-1](x)

        return x, h


class pose_prediction_rnn(nn.Module):
    def __init__(self, backbone = backbone_GRU(input_size=42, gru_hidden_size=168, gru_num_layers=1, fc_layers = [168, 42])):
        super(pose_prediction_rnn, self).__init__()
        self.backbone = backbone
        self.h_keeping = None
        self.ids_keeping = np.empty(0)

    def forward(self, x, ids=None, p=0.5, dropout = True, **kwargs):
        """
        params x : [b, f, j, 3]
        return : [b, j, 3]
        """
        h = self.get_h(ids)
        self.ids_keeping = ids
        b, f, j, _ = x.shape
        xyz_left = x[:,-1, 0]
        xyz_right = x[:,-1, 3]
        xyz_right_from_left = xyz_right - xyz_left
        theta = torch.atan2( xyz_right_from_left[...,1], xyz_right_from_left[...,0] )
        xyz_center = (xyz_left + xyz_right)/2
        cosTheta = torch.cos(theta[:,None,None])
        sinTheta = torch.sin(theta[:,None,None])
        R = torch.cat([torch.cat([cosTheta, sinTheta], -1), torch.cat([-sinTheta, cosTheta], -1)], -2)
        x = x - xyz_center[:,None,None]
        x[...,:2, None]  = R[:, None, None] @ x[...,:2, None] 
        x, self.h_keeping = self.backbone(x.view(1, b,-1), h, p = p, dropout = dropout)
        x = x.view(-1,14,3)
        x[...,:2, None]  = R.transpose(-1,-2)[:, None] @ x[...,:2, None]
        x = x + xyz_center[:,None]

        return x
        

    def get_h(self, ids):
        """
        params ids : [ int, ...]
        """
        if self.h_keeping is None or len(self.ids_keeping) == 0:
            return None
        _, N, h = self.h_keeping.shape
        num_sample = int(N/len(self.ids_keeping))
        h = torch.zeros(1, len(ids)*num_sample, h).to(self.h_keeping.device)
        ids_tracked = list(set(ids).intersection(set(self.ids_keeping)))
        idx1 = np.array([ids.index(id_) for id_ in ids_tracked])
        idx1 = (idx1[:,None]*num_sample + np.arange(num_sample)[None,:]).reshape(-1)
        idx2 = np.array([self.ids_keeping.index(id_) for id_ in ids_tracked])
        idx2 = (idx2[:,None]*num_sample + np.arange(num_sample)[None,:]).reshape(-1)
            
        h[:,idx1] = self.h_keeping[:,idx2]
        return h
    
    def resample(self, ids_pose, idxs_sample):
        """
        params idxs
        """
        if self.h_keeping is None or ids_pose is None or len(self.ids_keeping) == 0:
            return 

        n, s = idxs_sample.shape
        ids_tracked = list(set(ids_pose).intersection(set(self.ids_keeping)))
        ids1 = np.array([self.ids_keeping.index(id_) for id_ in ids_tracked])
        ids2 = np.array([ids_pose.index(id_) for id_ in ids_tracked])
        idxs1 = (ids1[:,None]*s + np.arange(s)[None,:]).reshape(-1)
        idxs2 = (ids1[:,None]*s + idxs_sample[ids2]).reshape(-1)
        
        self.h_keeping[:,idxs1] = self.h_keeping[:,idxs2]
        return