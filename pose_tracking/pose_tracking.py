import itertools
import json 
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.multivariate_normal import MultivariateNormal
from pose_prediction.setup import load_model
from view.openpose3d import eval_nppose3Ds_on_hm, eval_nppose3Ds_on_paf
from view.openpose3d import eval_ptpose3Ds_on_hm, eval_ptpose3Ds_on_paf

LIMIT_LIMB_LENGTH = np.array([1.2, 1.2, 0.8, 0.8, 0.6, 0.8, 0.8, 0.5, 0.8, 0.8, 0.5, 0.8, 0.8, 0.3])
LIMIT_LIMB_IDX = np.array([[3, 9], [6, 0], [3, 4], [4, 5], [3, 0], [0, 1], [1, 2], [12, 9], [9, 10], [10, 11], [12, 6], [6, 7], [7, 8], [12, 13]]).T
IDX_PAF = np.array([1, 2, 5, 6, 7, 8, 9, 11, 12, 13, 15])
IDX_PAF_JOINTS_FROM = np.array([3, 4, 0, 1, 12, 9, 10, 12, 6, 7, 12])
IDX_PAF_JOINTS_TO = np.array([4, 5, 1, 2, 9, 10, 11, 6, 7, 8, 13])

LIMB = np.array([ [0, 1], [1, 2], [0, 3], [3, 4], [4, 5], [0, 6], [6, 7], [7, 8], [3, 9], [9, 10], [10, 11], [9, 12], [12, 6], [12, 13] ]).T
LIMB_MEAN = torch.tensor([0.3982833  ,0.42993364 ,0.18530907 ,0.40203145 ,0.42922407 ,0.50275356
 ,0.2831661  ,0.19199172 ,0.49600458 ,0.2868699  ,0.19091912 ,0.18545817
 ,0.17876178 ,0.18985878])
LIMB_STD = torch.tensor([0.02977692 ,0.02476305 ,0.0397552  ,0.03180797 ,0.02559702 ,0.04293538
 ,0.02030252 ,0.02284359 ,0.04468039 ,0.0223102  ,0.02328202 ,0.02178835
 ,0.02514223 ,0.01745188])

class PoseTracking:
    def __init__(self, fullpath_json, n_camera, n_sample, device, n_random_search = 100):
        """
        param fullpath_json : str
        param n_camera : int
        param n_sample : int
        param device : torch.device()
        """
        self.R = n_random_search
        self.device = device 
        self.model_pc, self.model_pp = load_model(fullpath_json, device)
        
        self.v = n_camera
        self.s = n_sample
        self.J = 14
        self.P = 11
        pairs_views = torch.tensor(list(itertools.combinations(range(self.v),2)))
        self.vp = len(pairs_views)
        self.idx_s = torch.arange(self.s)[None, None, :, None, None].to(device)
        self.idx_P = torch.arange(self.P)[None, None, None, :, None].to(device)
        self.idx_v = pairs_views[None, :, None, None, :].to(device)
        self.idx_J = torch.arange(self.J)[None, None, None, None, :, None].to(device)
        self.idx_vp = torch.arange(self.vp)[None, :, None, None, None, None].to(device)
        self.idx_svp = torch.arange(int(self.s/self.vp))[None, None, :, None, None, None].to(device)
        self.idx_R = torch.arange(self.R)[None, None, None, :, None, None].to(device)
        self.LIMB_MEAN = LIMB_MEAN.to(device)
        self.LIMB_STD = LIMB_STD.to(device)
        
        self.sobel_x = torch.Tensor([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]]).view((1,1,3,3)).to(self.device)
        self.sobel_y = torch.Tensor([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]]).view((1,1,3,3)).to(self.device)

    def prediction(self, batch, p = 0.0, dropout = False, ids = None, **kwargs):
        """
        param batch : torch.tensor.device { n, s, j, 3 } 
        """
        if len(batch) > 0:
            n, s, j, _ = batch.shape
            return self.model_pp(batch.reshape(n*s, 1, j, 3), p = p, dropout = dropout, ids = ids).reshape(n, s, j, 3)
        else:
            return torch.zeros(0, self.s, 14, 3).to(self.device)


    def completion(self, clusters_simplified3Ds, p = 0.0, dropout = False):
        """
        param clusters_simplified3Ds : [ [ np.array{ 1 x 14 x 4}, ... ], ... ]
        return torch.tensor.device { len(clusters_simplified3Ds) * self.s x 1 x 14 x 3 } 
        """
        if len(clusters_simplified3Ds) > 0:
            batch_simplified3Ds = self.trans_clusters_to_batch(clusters_simplified3Ds)
            batch_simplified3Ds = torch.tensor(batch_simplified3Ds, dtype=torch.float).to(self.device)
            return self.model_pc(batch_simplified3Ds, p = p, dropout = dropout)
        else:
            return torch.zeros(0, self.s, 14, 3).to(self.device)


    def update(self, batch_simpl3Ds, ids, idxs_op_to_simple, hmss, pafss, camera_matrixs, std_joint = 0.03, eps = 0.1):
        """
        param batch_simpl3Ds : torch.tensor.device { n x s x J x 3 }
        param idxs_op_to_simple : [ idx, ... ]
        param hmss : torch.tensor.device { v x J x h x w }
        param pafss : torch.tensor.device { v x 2*P x h x w }
        param camera_matrixs : torch.tensor.device { v x 3 x 4 }
        """
        if len(batch_simpl3Ds) == 0:
            return batch_simpl3Ds
        batch_simpl3Ds = self.update_pose_resampling(batch_simpl3Ds, ids, idxs_op_to_simple, hmss, pafss, camera_matrixs, eps)
        batch_simpl3Ds = self.update_joint_random_search(batch_simpl3Ds, idxs_op_to_simple, hmss, camera_matrixs, std_joint, eps)        
        return batch_simpl3Ds


    def update_pose_resampling(self, batch_simpl3Ds, ids, idxs_op_to_simple, hmss, pafss, camera_matrixs, eps = 0.1):
        n, s, J, _ = batch_simpl3Ds.shape
        idx_n = torch.arange(n)[:, None, None]
        
        hm_batch = eval_ptpose3Ds_on_hm(batch_simpl3Ds.view(-1,J,3), idxs_op_to_simple, hmss, camera_matrixs)
        hm_batch = hm_batch.reshape(n, s, J, len(hmss))
        paf_batch = eval_ptpose3Ds_on_paf(batch_simpl3Ds.reshape(-1,*batch_simpl3Ds.shape[-2:]), idxs_op_to_simple, pafss, camera_matrixs)
        paf_batch[paf_batch < 0.0] = 0.0
        paf_batch = paf_batch.reshape(n, s, 11, len(hmss)) + eps
        paf_batch *= (hm_batch[...,IDX_PAF_JOINTS_FROM, :] + eps) * (hm_batch[...,IDX_PAF_JOINTS_TO, :] + eps)
        
        paf_batch_on_each_pairs = paf_batch[idx_n[..., None, None], self.idx_s, self.idx_P, self.idx_v] 
        
        w = (paf_batch_on_each_pairs.norm(dim=-1) + eps).prod(-1)
        w = w.reshape(n*self.vp,s)
        idxs = self.resample_idx(w, int(self.s/self.vp)).reshape(n, -1).cpu().numpy()
        self.model_pp.resample(ids_pose = ids, idxs_sample = idxs)
        batch_simpl3Ds = batch_simpl3Ds[np.arange(n)[:,None], idxs] # n x s x ...   
        
        return batch_simpl3Ds

    def update_joint_random_search(self, batch_simpl3Ds, idxs_op_to_simple, hmss, camera_matrixs, std_joint = 0.03, eps = 0.1):
        n, s, J, _ = batch_simpl3Ds.shape
        for i in range(n):
            batch_simpl3Ds_random = batch_simpl3Ds[i, :, None] + torch.zeros(self.s, self.R, self.J, 3, device=self.device).normal_(0.0, std_joint)
            hm_batch = eval_ptpose3Ds_on_hm(batch_simpl3Ds_random.view(-1,J,3), idxs_op_to_simple, hmss, camera_matrixs)
            hm_batch = hm_batch.view(self.vp, int(s/self.vp), self.R, J, len(hmss))
            hm_batch_on_each_pairs = hm_batch[self.idx_vp, self.idx_svp, self.idx_R, self.idx_J, self.idx_v[..., None, :]]
            
            w = (hm_batch_on_each_pairs + eps).prod(-1)
            idx_r = w.argmax(-2).reshape(s,J) 
            idx_s = self.idx_s[0,0,...,0]
            idx_J = self.idx_J[0,0,0,...,0]
            batch_simpl3Ds[i] = batch_simpl3Ds_random[idx_s, idx_r, idx_J]
        
        return batch_simpl3Ds


    def update_joint_gradient_ascent(self, batch_simpl3Ds, idxs_op_to_simple, hmss, camera_matrixs, iter_refine):
        n, s, J, _ = batch_simpl3Ds.shape
        v, q, h, w = hmss.shape
        hmss_pad = F.pad(hmss.view(-1,1,h,w), pad=(1,1,1,1), mode='replicate')
        
        G_x = F.conv2d(hmss_pad, self.sobel_x, stride=1).view(v,q,h,w)
        G_y = F.conv2d(hmss_pad, self.sobel_y, stride=1).view(v,q,h,w)
        
        for i in range(iter_refine):
            xyz1 = torch.ones(*batch_simpl3Ds.shape[:-1], 4).to(self.device)
            xyz1[...,:-1] = batch_simpl3Ds
            xyz1 = xyz1.reshape(n, self.vp, int(self.s/self.vp), 1, J, 4)
            xy1 = (camera_matrixs[self.idx_v[:,:,0,:,:,None]] @ xyz1[...,None]).squeeze(-1)
            z = xy1[...,-1]
            xy1 /= z[...,None]
            xy = xy1[...,:2]
            xy_long = torch.round(xy).long()
            bool_in = ((xy_long >= 0) & (xy_long < torch.tensor([w, h], dtype=torch.long).to(self.device))).all(-1)
            
            idx_v = self.idx_v[...,0,:,None].repeat(n,1,int(self.s/self.vp),1,J)[bool_in]
            idx_j = torch.tensor(idxs_op_to_simple)[None,None,None,None,:].repeat(n,self.vp,int(self.s/self.vp),2,1)[bool_in]
            idx_h = xy_long[...,1][bool_in].long()
            idx_w = xy_long[...,0][bool_in].long()
            
            G_x_batch = torch.zeros(n, self.vp, int(self.s/self.vp), 2, J).to(self.device)
            G_x_batch[bool_in] = G_x[idx_v, idx_j, idx_h, idx_w]
            G_y_batch = torch.zeros(n, self.vp, int(self.s/self.vp), 2, J).to(self.device)
            G_y_batch[bool_in] = G_y[idx_v, idx_j, idx_h, idx_w]
            G_xy_batch = torch.cat((G_x_batch[...,None,None], G_y_batch[...,None,None]), dim=-1)
            
            A = xy[...,None] @ camera_matrixs[self.idx_v[:,:,0,:,:,None], -1:,:3] - camera_matrixs[self.idx_v[:,:,0,:,:,None], :2,:3]
            A /= A.norm(dim=-1,keepdim=True)
            A *= z[...,None,None]
            dX = (G_xy_batch @ A).mean(dim=(-4))[...,0,:].reshape(*batch_simpl3Ds.shape)
            batch_simpl3Ds += 0.1*0.1**(0.05*i)*dX
            
            hm_batch = eval_ptpose3Ds_on_hm(batch_simpl3Ds.view(-1,J,3), idxs_op_to_simple, hmss, camera_matrixs)
            hm_batch = hm_batch.reshape(n, s, J, len(hmss))
        
        return batch_simpl3Ds


    def resample_idx(self, w, n_resample = None):
        """
        param w : np.array { n x s }
        return idxs : np.array { n x s x 14 x 3 } 
        """
        if n_resample is None:
            n_resample = self.s
        n, s = w.shape
        n_poses = len(w)
        w_sum = w.sum(-1)
        idx_n = w_sum != 0.0
        n_valid = len(idx_n)
        w[idx_n] /= w_sum[idx_n,None]
        c = torch.cumsum(w[idx_n],dim=1)
        c /= c[...,-1,None]
        r = torch.FloatTensor(n_valid, 1).uniform_(0, 1./n_resample)
        u = ((torch.arange(n_resample, dtype=torch.float)/n_resample)[None,:] + r).to(w.device)
        idxs = (c[:,None,:] < u[:,:,None]).sum(-1)
        return idxs


    def trans_clusters_to_batch(self, clusters_simplified3Ds):
        """
        param clusters_simplified3Ds : [ [ np.array{ 1 x 14 x 4}, ... ], ... ]
        return np.array { len(clusters_simplified3Ds) * self.s x 1 x 14 x 4 } 
        """
        batch_simplified3Ds = []
        for cluster_simplified3Ds in clusters_simplified3Ds:
            batch_cluster = np.array(cluster_simplified3Ds)
            b = batch_cluster.shape[0]
            r = int(self.s/b)    
            batch_simplified3Ds.append(np.concatenate((np.tile(batch_cluster,(r,1,1)), batch_cluster[:self.s - r*b]), axis=0))
        return np.array(batch_simplified3Ds)


def get_joint_means_from_batch(batch_simpl3Ds, idxs_op_to_simple, hmss, pafss, camera_matrixs):
    """
    """
    n, s, J, _ = batch_simpl3Ds.shape
    w = eval_joint(batch_simpl3Ds.cpu().numpy(), idxs_op_to_simple, hmss, pafss, camera_matrixs)
    w = w.swapaxes(1, 2)
    n_poses = len(w)
    w_sum = w.sum(-1)
    idx_n, idx_j = np.where(w_sum != 0.0)
    w[idx_n, idx_j] /= w_sum[idx_n, idx_j, None]
    idx_n, idx_j = np.where(w_sum == 0.0)
    w[idx_n, idx_j] = 1.0 / s
    simpl3Ds_mean = batch_simpl3Ds * torch.tensor(w.swapaxes(1, 2)[...,None], dtype=torch.float).to(batch_simpl3Ds.device)
    simpl3Ds_mean = simpl3Ds_mean.sum(1)
    return simpl3Ds_mean


def eval_pose(batch_simpl3Ds, idxs_op_to_simple, hmss, pafss, camera_matrixs, score_pair = 1.0, cnts_overlapping = None, limit_limb = True):
    """
    param batch_simpl3Ds : torch.tensor.device { n x s x J x 3 }
    param idxs_op_to_simple : [ idx, ... ]
    param hmss : np.array { v x J x h x w }
    param pafss : np.array { v x 2*P x h x w }
    param camera_matrixs : np.array{ v x 3 x 4 }
    """
    hmss = np.copy(hmss)
    hmss[:,0] = np.max(hmss[:,(0,-4,-3,-2,-1)], axis=1)
    hm_batch = eval_nppose3Ds_on_hm(batch_simpl3Ds.reshape(-1,*batch_simpl3Ds.shape[-2:]), idxs_op_to_simple, hmss, camera_matrixs)
    if isinstance(cnts_overlapping, np.ndarray):
        idx_n, idx_j, idx_v = np.where(cnts_overlapping > 1.0)
        hm_batch[idx_n, idx_j, idx_v] /= cnts_overlapping[idx_n, idx_j, idx_v]

    paf_batch = eval_nppose3Ds_on_paf(batch_simpl3Ds.reshape(-1,*batch_simpl3Ds.shape[-2:]), idxs_op_to_simple, pafss, camera_matrixs)
    paf_batch *= (hm_batch[:,IDX_PAF_JOINTS_FROM]+0.1) * (hm_batch[:,IDX_PAF_JOINTS_TO]+0.1)
    
    w = (np.linalg.norm(paf_batch, axis=-1) + 0.1).prod(-1).reshape(batch_simpl3Ds.shape[:-2])
    
    if limit_limb:
        idx_over_limb_limit = np.any(np.linalg.norm(batch_simpl3Ds[...,LIMIT_LIMB_IDX[0],:] - batch_simpl3Ds[...,LIMIT_LIMB_IDX[1],:], axis = -1) > LIMIT_LIMB_LENGTH[None,:], axis = -1)
        w[idx_over_limb_limit] = 0
    
    return w
