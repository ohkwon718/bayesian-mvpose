import torch
import torch.nn as nn
import torch.nn.functional as F

def get_pose_completion(device):
    layers = [56, 2048, 2048, 2048, 42]
    pose_completion = PoseCompletion(fc_dropout(layers)).to(device)
    state = torch.load("pose_prediction/model_completion_fc.pt")
    pose_completion.load_state_dict(state['model_state_dict'])
    pose_completion.eval()
    return pose_completion


class fc_dropout(nn.Module):
    def __init__(self, D = [168, 336, 42], act = F.relu, bn = False):
        super(fc_dropout, self).__init__()
        self.act = act
        self.bn = bn
        self.fcs = []
        self.bns = []
        for i in range(len(D)-2):
            self.fcs.append(nn.Linear(D[i], D[i+1]))
            self.add_module("fc{:02d}"+str(i), self.fcs[-1])
            if self.bn:
                self.bns.append(nn.BatchNorm1d(D[i+1]))
                self.add_module("bn{:02d}"+str(i), self.bns[-1])
            else:
                self.bns.append(lambda x: x)
        self.fcs.append(nn.Linear(D[-2], D[-1]))
        self.add_module("fc{:02d}"+str(len(D)-2), self.fcs[-1])
            
    def forward(self, x, p=0.5, dropout = True):
        for i in range(len(self.fcs)-1):
            fc = self.fcs[i]
            bn = self.bns[i]
            x = self.act(bn(fc(x)))
            x = F.dropout(x, p=p, training = dropout)
        x = self.fcs[-1](x)
        return x


class PoseCompletion(nn.Module):
    def __init__(self, backbonn = fc_dropout([56, 2048, 2048, 2048, 42])):
        super(PoseCompletion, self).__init__()
        self.backbonn = backbonn

    def forward(self, x, p=0.5, dropout = True, **kwargs):
        """
        param x : [n, s, j, 4]
        return : [n, s, j, 3]
        """
        n, s, j, _ = x.shape
        x = x.view(n*s,j,4)
        x_ = x.sum(dim=1)
        x_ /= x_[...,3:4]
        x = x.clone()
        x[...,:3] -= x_[..., None,:3]
        x[...,:3] *= x[...,3:4]
        x = self.backbonn(x.view(n*s,-1), p=p, dropout = dropout)
        x = x.view(n, s, j, 3) + x_.view(n, s, 1, 4)[..., :3]
        return x
