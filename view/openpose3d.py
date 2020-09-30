from __future__ import print_function

import os
import sys
import copy
import numpy as np
import torch
import scipy 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

from skimage.feature import peak_local_max
from scipy.cluster.hierarchy import linkage, fcluster

from view.stereo import get_3d_points_from_multiple_observation


IDXS_PAF_TO_JOINT = [(0,1,8),(1,9,10),(2,10,11),(3,8,9),(4,8,12),(5,12,13),(6,13,14),(7,1,2),
                (8,2,3),(9,3,4),(10,2,17),(11,1,5),(12,5,6),(13,6,7),(14,5,18),(15,1,0),
                (16,0,15),(17,0,16),(18,15,17),(19,16,18)]


DICT_JOINT_TO_PAF = {(1, 8): (0, False), (8, 1): (0, True), (9, 10): (1, False), (10, 9): (1, True), 
                (10, 11): (2, False), (11, 10): (2, True), (8, 9): (3, False), (9, 8): (3, True), 
                (8, 12): (4, False), (12, 8): (4, True), (12, 13): (5, False), (13, 12): (5, True), 
                (13, 14): (6, False), (14, 13): (6, True), (1, 2): (7, False), (2, 1): (7, True), 
                (2, 3): (8, False), (3, 2): (8, True), (3, 4): (9, False), (4, 3): (9, True), 
                (2, 17): (10, False), (17, 2): (10, True), (1, 5): (11, False), (5, 1): (11, True), 
                (5, 6): (12, False), (6, 5): (12, True), (6, 7): (13, False), (7, 6): (13, True), 
                (5, 18): (14, False), (18, 5): (14, True), (1, 0): (15, False), (0, 1): (15, True), 
                (0, 15): (16, False), (15, 0): (16, True), (0, 16): (17, False), (16, 0): (17, True), 
                (15, 17): (18, False), (17, 15): (18, True), (16, 18): (19, False), (18, 16): (19, True) }

LIMIT_LIMB_LENGTH = [1.2, 0.8, 0.8, 0.3, 0.3, 0.8, 0.8, 0.5, 0.8, 0.8, 
                    0.6, 0.5, 0.8, 0.8, 0.6, 0.3, 0.3, 0.3, 0.3, 0.3]


def get_pose3Ds_from_multi_hms(hmss, pafss, camera_matrixs, fundamental_matrixs,
                                thold_paf = 0.7, 
                                min_num_of_joint = 5, thold_cluster = 0.05,
                                thold_hm_peak=0.3, thold_sv = 15, num_sample_paf = 5):
    """
    param hmss : np.array { v x J x h x w }
    param pafss : np.array { v x 2*P x h x w }
    param camera_matrixs : np.array{ v x 3 x 4 }
    return [ dict{0:np.array{3}, 1:np.array{3}, ...}, ...]
    """
    
    joint3d_candidates = []
    joint2d_candidates = []
    
    for i in range(19):
        hms = hmss[:,i,...]
        candidate_sets = [ peak_local_max( hms[i], min_distance=5, threshold_abs=thold_hm_peak)[:,[1,0]].tolist() for i in range(len(hms))]
        joint2d_candidates.append(copy.deepcopy(candidate_sets))
        joint_candidate = get_3d_points_from_multiple_observation(candidate_sets, camera_matrixs, fundamental_matrixs, thold_sv)
        if len(joint_candidate) >= 2:
            joint_candidate = np.array(joint_candidate)
            Z = linkage(joint_candidate, 'ward')
            cluster = fcluster(Z, t=thold_cluster, criterion='distance')
            cluster_mean = [joint_candidate[cluster == c+1].mean(axis=0) for c in range(cluster.max())]
            joint3d_candidates.append( cluster_mean )
        else:
            joint3d_candidates.append(joint_candidate)
        
    pose3Ds = build_pose3Ds_from_candidates(joint3d_candidates, pafss, camera_matrixs, thold_paf, num_sample_paf)
    return [pose3D for pose3D in pose3Ds if len(pose3D) >= min_num_of_joint], joint3d_candidates, joint2d_candidates
    

def build_pose3Ds_from_candidates(joint3d_candidates, pafss, camera_matrixs, thold_paf = 0.7, num_sample_paf = 10):
    """
    param joint3d_candidates : [ [ np.array{3}, ... ], ... ]
    param pafss : np.array { v x 2*P x h x w }
    param camera_matrixs : np.array{ v x 3 x 4 }
    return [ dict{0:np.array{3}, 1:np.array{3}, ...}, ...]
    """
    pose3Ds = []
    for i_j, joint_candidate in enumerate(joint3d_candidates):       
        residue = assign_candidates_to_pose3Ds(joint_candidate, i_j, pose3Ds, pafss, camera_matrixs, 
                                                thold_paf, num_sample_paf) 
        for xyz in residue:
            pose3Ds.append({i_j:xyz})

    for i_j, joint_candidate in enumerate(reversed(joint3d_candidates)):       
        assign_candidates_to_pose3Ds(joint_candidate, i_j, pose3Ds, pafss, camera_matrixs, 
                                    thold_paf, num_sample_paf) 
    
    return pose3Ds

def assign_candidates_to_pose3Ds(joint_candidate, joint, pose3Ds, pafss, camera_matrixs, thold_paf = 0.7, num_sample_paf = 10):
    """
    param joint_candidate : [ np.array{3}, ... ]
    param joint : int
    param pose3Ds : dict{0:np.array{3}, 1:np.array{3}, ...}
    param pafss : np.array { v x 2*P x h x w }
    param camera_matrixs : np.array{ v x 3 x 4 }
    return [ dict{0:np.array{3}, 1:np.array{3}, ...}, ...]
    """
    
    idx_pose3D_valid = [i for i, pose3D in enumerate(pose3Ds) if joint not in pose3D]
    idx_pafs = []
    xyz_from = []
    xyz_to = []
    idx_batchs = [0]
    cost_paf = np.zeros((len(idx_pose3D_valid), len(joint_candidate)))
    
    valid_candidates = []
    idx_valid_candidates = []

    for i_pose3D, idx_pose3D in enumerate(idx_pose3D_valid):
        pose3D = pose3Ds[idx_pose3D]
        joint_candidate_on_pose3D = []
        num_valid_candidate = 0
        # valid_candidate = []
        idx_valid_candidate = []

        for i_candidate, xyz in enumerate(joint_candidate):
            if all([(key, joint) not in DICT_JOINT_TO_PAF or np.linalg.norm(xyz - pose3D[key]) <  LIMIT_LIMB_LENGTH[DICT_JOINT_TO_PAF[(key, joint)][0]] for key in pose3D]):
                num_valid_paf = 0
                for key in pose3D.keys():
                    if (key, joint) in DICT_JOINT_TO_PAF:
                        idx_paf, is_negative = DICT_JOINT_TO_PAF[(key, joint)]
                        idx_pafs.append(idx_paf)
                        if is_negative:
                            xyz_from.append(xyz)
                            xyz_to.append(pose3D[key])
                        else:
                            xyz_from.append(pose3D[key])
                            xyz_to.append(xyz)
                        num_valid_paf += 1
                idx_batchs.append(num_valid_paf)
                idx_valid_candidate.append(i_candidate)
        idx_valid_candidates.append(idx_valid_candidate)

    num_valid_candidates = [len(idx_valid_candidate) for idx_valid_candidate in idx_valid_candidates]
    
    if len(idx_pafs) > 0:
        idx_pafs = np.array(idx_pafs)
        xyz_from = np.array(xyz_from)
        xyz_to = np.array(xyz_to)
        integrate_paf = integrate_paf_along_two_3d_nppoints(xyz_from, xyz_to, idx_pafs, pafss, camera_matrixs)
        
        idx_batchs = np.array(idx_batchs)
        idx_batchs = np.cumsum(idx_batchs)
        cost_paf = [-np.linalg.norm(integrate_paf[idx_batchs[i]:idx_batchs[i+1]]) for i in range(len(idx_batchs) - 1)]
        cost_paf = [cost_paf[sum(num_valid_candidates[:i]):sum(num_valid_candidates[:i+1])] for i in range(len(num_valid_candidates))]
    
    candidates_assigned = []

    for i, paf_row in enumerate(cost_paf):
        if len(paf_row) > 0:
            j = np.argmin(paf_row)
            if paf_row[j] < -thold_paf:
                pose3Ds[idx_pose3D_valid[i]][joint] = joint_candidate[idx_valid_candidates[i][j]]
                candidates_assigned.append(idx_valid_candidates[i][j])
    
    residue = [xyz for i_candidate, xyz in enumerate(joint_candidate) if not i_candidate in candidates_assigned]
    return residue



def eval_nppose3Ds(pose3Ds, idxs_part, hmss, pafss, camera_matrixs):
    """
    param pose3Ds : np.array{ n x 14 x 3 }
    param idxs_part : [ int, ...]
    param hmss : np.array { v x J x h x w }
    param pafss : np.array { v x 2*P x h x w }
    param camera_matrixs : np.array{ v x 3 x 4 }
    return hms_pose : [np.array{ n x 14 x v }, ...]
    return pafs_pose : [np.array{ n x 11 x v }, ...]
    """
    return eval_nppose3Ds_on_hm(pose3Ds, idxs_part, hmss, camera_matrixs), eval_nppose3Ds_on_paf(pose3Ds, idxs_part, pafss, camera_matrixs)
    

def eval_nppose3Ds_on_hm(pose3Ds, idxs_part, hmss, camera_matrixs):
    """
    param pose3Ds : np.array{ n x 14 x 3 }
    param idxs_part : [ int, ...]
    param hmss : np.array { v x J x h x w }
    param camera_matrixs : np.array{ v x 3 x 4 }
    return hms_pose : [np.array{ n x 14 x v }, ...]
    """
    n = len(pose3Ds)
    hm_poses = get_hm_on_projected_3d_nppoints(pose3Ds.reshape(-1, 3), np.tile(idxs_part, n), hmss, camera_matrixs)
    hm_poses = hm_poses.reshape(n, 14, len(camera_matrixs))
    
    return hm_poses


def eval_nppose3Ds_on_paf(pose3Ds, idxs_part, pafss, camera_matrixs):
    """
    param pose3Ds : np.array{ n x 14 x 3 }
    param idxs_part : [ int, ...]
    param pafss : np.array { v x 2*P x h x w }
    param camera_matrixs : np.array{ v x 3 x 4 }
    return pafs_pose : [np.array{ n x 11 x v }, ...]
    """
    n = len(pose3Ds)
    idx_valid_paf, idx_joints_from, idx_joints_to = get_valid_paf_idxs_from_joints(np.array(idxs_part))
    paf_poses = integrate_paf_along_two_3d_nppoints(pose3Ds[:, idx_joints_from].reshape(-1,3), 
                                                pose3Ds[:, idx_joints_to].reshape(-1,3),
                                                np.tile(idx_valid_paf, n), pafss, camera_matrixs, n_sample=3)
    paf_poses = paf_poses.reshape(n, 11, len(camera_matrixs))

    return paf_poses

def get_hm_on_projected_3d_nppoints(xyz, idx_hm, hmss, camera_matrixs):
    """
    param xyz : np.array{ n x 3 }
    param idx_hm : np.array{ n }
    param hmss : np.array { v x J x h x w }
    param camera_matrixs : np.array{ v x 3 x 4 }
    return np.array { n x v }
    """
    v, _, h, w = hmss.shape
    n = len(idx_hm)
    
    xyz1 = np.ones((n, 4))
    xyz1[:,:-1] = xyz

    xy1 = (xyz1[:,None,None,...] @ np.swapaxes(camera_matrixs, -1, -2)).squeeze(-2)
    xy1 /= xy1[...,-1:]
    xy = np.around(xy1[...,:2]).astype(int)
    bool_out = np.any( np.logical_or(xy < 0, xy >= np.array((w,h))) , axis=-1)
    bool_in = np.logical_not(bool_out)

    idx_v = np.tile(np.arange(v), (n,1))[bool_in]
    idx_j = np.tile(idx_hm[:,None], (1,v))[bool_in]
    idx_h = xy[bool_in,1]
    idx_w = xy[bool_in,0]
    
    hm_pose = np.zeros((n, v))
    hm_pose[bool_in] = hmss[idx_v, idx_j, idx_h, idx_w]

    return hm_pose


def integrate_paf_along_two_3d_nppoints(xyz_from, xyz_to, idx_paf, pafss, camera_matrixs, n_sample=10):
    """
    param xyz_from : np.array{ n x 3 }
    param xyz_to : np.array{ n x 3 }
    param idx_paf : np.array{ n }
    param pafss : np.array { v x 2*P x h x w }
    param camera_matrixs : np.array{ v x 3 x 4 }
    return np.array { n x v }
    """
    n = len(xyz_from)
    v, _, h, w = pafss.shape
    xyz1_from = np.ones((n, 4))
    xyz1_from[:,:-1] = xyz_from
    xyz1_to = np.ones((n, 4))
    xyz1_to[:,:-1] = xyz_to

    xyz1_sample = np.linspace( xyz1_from, xyz1_to, num=n_sample+2, axis=1)[...,1:-1,:]
    xy1_sample = xyz1_sample[:,None,...] @ np.swapaxes(camera_matrixs, -1, -2)
    xy1_sample /= xy1_sample[...,-1:]
    xy_sample = np.around(xy1_sample[...,:2]).astype(int)
    bool_out = np.any( np.logical_or(xy_sample < 0, xy_sample >= np.array((w,h))) , axis=-1)
    bool_in = np.logical_not(bool_out)

    num_valid = bool_in.sum(-1)
    vec = xy1_sample[...,-1,:2] - xy1_sample[...,0,:2]
    vec_norm = np.linalg.norm(vec, axis=-1)
    idx = vec_norm != 0
    vec[idx] /= vec_norm[idx, None]
    vec = np.tile(vec[...,None,:], (1, 1, n_sample, 1))[bool_in]
    
    idx_v = np.tile(np.arange(v)[None,:,None], (n,1,n_sample))[bool_in,None]
    idx_p = np.tile(np.column_stack([idx_paf*2, idx_paf*2+1])[:,None,None,:], (1,v,n_sample,1))[bool_in]
    idx_h = xy_sample[bool_in,1,None]
    idx_w = xy_sample[bool_in,0,None]
    paf_sample = np.zeros((n, v, n_sample))
    paf_sample[bool_in] = (pafss[idx_v,idx_p,idx_h,idx_w] * vec).sum(1)
    paf_int = paf_sample.mean(-1)
    return paf_int


def get_valid_paf_idxs_from_joints(idx_joints):
    """
    param idx_joints : np.array{ n } which is a list of joint indices
    return idx_valid_paf : np.array{ m } 
    return idx_joints_from : np.array{ m } 
    return idx_joints_to : np.array{ m } 
    """

    idx_valid = np.array([idx_paf for idx_paf in IDXS_PAF_TO_JOINT if idx_paf[1] in idx_joints and idx_paf[2] in idx_joints])
    idx_joints_sorted = np.argsort(idx_joints)
    
    return idx_valid[:,0], idx_joints_sorted[np.searchsorted(idx_joints[idx_joints_sorted], idx_valid[:,1])], idx_joints_sorted[np.searchsorted(idx_joints[idx_joints_sorted], idx_valid[:,2])]



###################################
#              pytorch            #
###################################

def eval_ptpose3Ds_on_hm(pose3Ds, idxs_part, hmss, camera_matrixs):
    """
    param pose3Ds : torch.tensor{ n x 14 x 3 }
    param idxs_part : [ int, ...]
    param hmss : torch.tensor { v x J x h x w }
    param camera_matrixs : torch.tensor{ v x 3 x 4 }
    return hms_pose : [torch.tensor{ n x 14 x v }, ...]
    """
    n = len(pose3Ds)
    hm_poses = get_hm_on_projected_3d_ptpoints(pose3Ds.view(-1, 3), np.tile(idxs_part, n), hmss, camera_matrixs)
    hm_poses = hm_poses.view(n, 14, len(camera_matrixs))
    
    return hm_poses


def eval_ptpose3Ds_on_paf(pose3Ds, idxs_part, pafss, camera_matrixs):
    """
    param pose3Ds : torch.tensor{ n x 14 x 3 }
    param idxs_part : [ int, ...]
    param pafss : torch.tensor { v x 2*P x h x w }
    param camera_matrixs : torch.tensor{ v x 3 x 4 }
    return pafs_pose : [torch.tensor{ n x 11 x v }, ...]
    """
    n = len(pose3Ds)
    idx_valid_paf, idx_joints_from, idx_joints_to = get_valid_paf_idxs_from_joints(np.array(idxs_part))
    paf_poses = integrate_paf_along_two_3d_ptpoints(pose3Ds[:, idx_joints_from].reshape(-1,3), 
                                                pose3Ds[:, idx_joints_to].reshape(-1,3),
                                                np.tile(idx_valid_paf, n), pafss, camera_matrixs, n_sample=3)
    paf_poses = paf_poses.reshape(n, 11, len(camera_matrixs))

    return paf_poses

def get_hm_on_projected_3d_ptpoints(xyz, idx_hm, hmss, camera_matrixs):
    """
    param xyz : torch.tensor { n x 3 }
    param idx_hm : np.array { n }
    param hmss : torch.tensor { v x J x h x w }
    param camera_matrixs : torch.tensor{ v x 3 x 4 }
    return torch.tensor { n x v }
    """
    device = xyz.device
    v, _, h, w = hmss.shape
    n = len(idx_hm)
    
    xyz1 = torch.ones(n, 4).to(device)
    xyz1[:,:-1] = xyz
    
    xy1 = (xyz1[:,None,None,...] @ camera_matrixs.transpose(-1,-2)).squeeze(-2)
    xy1 /= xy1[...,-1:]
    xy = torch.round(xy1[...,:2]).long()
    
    bool_in = ((xy >= 0) & (xy < torch.tensor([w, h], dtype=torch.long).to(device))).all(-1)
    
    idx_v = torch.arange(v).repeat(n,1)[bool_in]
    idx_j = torch.from_numpy(idx_hm[:,None]).repeat(1,v)[bool_in]
    idx_h = xy[...,1][bool_in].long()
    idx_w = xy[...,0][bool_in].long()
    
    hm_pose = torch.zeros(n, v).to(device)
    hm_pose[bool_in] = hmss[idx_v, idx_j, idx_h, idx_w]

    return hm_pose


def integrate_paf_along_two_3d_ptpoints(xyz_from, xyz_to, idx_paf, pafss, camera_matrixs, n_sample=10):
    """
    param xyz_from : torch.tensor { n x 3 }
    param xyz_to : torch.tensor { n x 3 }
    param idx_paf : torch.tensor { n }
    param pafss : torch.tensor  { v x 2*P x h x w }
    param camera_matrixs : torch.tensor { v x 3 x 4 }
    return torch.tensor { n x v }
    """
    assert xyz_from.device == xyz_to.device
    device = xyz_from.device
    n = len(xyz_from)
    v, _, h, w = pafss.shape
    
    xyz1_from = torch.ones(n, 4).to(device)
    xyz1_from[:,:-1] = xyz_from
    xyz1_to = torch.ones(n, 4).to(device)
    xyz1_to[:,:-1] = xyz_to

    ### pytorch should have multidimensional linspace ###
    xyz1_sample = xyz1_from[:,None,:] + (xyz1_to - xyz1_from)[:,None,:] * (torch.arange(1,n_sample+1,dtype=torch.float).to(device)/(n_sample+1))[None,:,None]
    xy1_sample = xyz1_sample[:,None,...] @ camera_matrixs.transpose(-1, -2)
    xy1_sample /= xy1_sample[...,-1:]
    xy_sample = torch.round(xy1_sample[...,:2]).long()
    bool_in = ((xy_sample >= 0) & (xy_sample < torch.tensor([w, h], dtype=torch.long).to(device))).all(-1)

    vec = xy1_sample[...,-1,:2] - xy1_sample[...,0,:2]
    vec_norm = vec.norm(p=2, dim=-1) 
    idx = vec_norm != 0
    vec[idx] /= vec_norm[idx][:, None]
    vec = vec[...,None,:].repeat(1, 1, n_sample, 1)[bool_in]
    
    idx_v = torch.arange(v)[None,:,None].repeat(n,1,n_sample)[bool_in][...,None]
    idx_p = torch.from_numpy(np.tile(np.column_stack([idx_paf*2, idx_paf*2+1])[:,None,None,:], (1,v,n_sample,1)))[bool_in]
    idx_h = xy_sample[bool_in][...,1,None]
    idx_w = xy_sample[bool_in][...,0,None]
    paf_sample = torch.zeros((n, v, n_sample)).to(device)
    paf_sample[bool_in] = (pafss[idx_v,idx_p,idx_h,idx_w] * vec).sum(1)
    paf_int = paf_sample.mean(-1)
    return paf_int

