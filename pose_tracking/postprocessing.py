import os
import json
from collections import OrderedDict
import numpy as np
import torch
from pose_models.pose_utils import get_trasforming_indices, trans_np_to_list, trans_list_to_np
import pose_models.simplified as pose_simpl
import pose_models.coco as pose_coco
import pose_models.openpose as pose_op
from pose_tracking.pose_tracking import eval_pose, get_joint_means_from_batch
from view.openpose3d import eval_ptpose3Ds_on_hm, eval_ptpose3Ds_on_paf

IDXS_SIMPL_TO_COCO = get_trasforming_indices(pose_simpl.JOINT_NAMES, pose_coco.JOINT_NAMES)
IDXS_COCO_TO_SIMPL = get_trasforming_indices(pose_coco.JOINT_NAMES, pose_simpl.JOINT_NAMES)
IDXS_OP_TO_SIMPL = get_trasforming_indices(pose_op.JOINT_NAMES, pose_simpl.JOINT_NAMES)
IDX_PAF_JOINTS_FROM = np.array([3, 4, 0, 1, 12, 9, 10, 12, 6, 7, 12])
IDX_PAF_JOINTS_TO = np.array([4, 5, 1, 2, 9, 10, 11, 6, 7, 8, 13])

class Tracker(object):
    def __init__(self, thold_trigger, thold_kill, n_frames_keep, camera_matrixs, scale = 1.0, axis = np.array([[0,1,2], [1,1,1]])):
        self.idx = 0
        self.thold_trigger = thold_trigger
        self.thold_kill = thold_kill
        self.n_frames_keep = n_frames_keep
        self.ids = []
        self.frames = []
        self.poses = []
        self.ws = []
        self.cnts_frames_triggered = []
        self.camera_matrixs = camera_matrixs
        self.tf_matrixs_axis_scale = np.diag(scale*axis[-1])[axis[0]]
        self.pose3Ds = []

    def update(self, frame, ids, pose3Ds, idxs_op_to_simple, hmss, pafss, camera_matrixs, eps = 0.1):
        """
        """
        n, s, J, _ = pose3Ds.shape
        if n == 0:
            return ids, pose3Ds, np.zeros((0,J,3))
        hm_batch = eval_ptpose3Ds_on_hm(pose3Ds.view(-1,*pose3Ds.shape[-2:]), idxs_op_to_simple, hmss, camera_matrixs)
        hm_batch = hm_batch.view(*pose3Ds.shape[:2], *hm_batch.shape[-2:])
        paf_batch = eval_ptpose3Ds_on_paf(pose3Ds.reshape(-1,*pose3Ds.shape[-2:]), idxs_op_to_simple, pafss, camera_matrixs)
        paf_batch[paf_batch < 0.0] = 0.0
        paf_batch = paf_batch.view(*pose3Ds.shape[:2], *paf_batch.shape[-2:]) + eps
        paf_batch *= (hm_batch[...,IDX_PAF_JOINTS_FROM, :]+eps) * (hm_batch[...,IDX_PAF_JOINTS_TO, :]+eps)
        w_pose = (paf_batch.norm(dim=-1) + eps).prod(-1)
        
        w_joint = (hm_batch + eps).prod(-1) * w_pose[...,None]
        w_joint = w_joint.transpose(1, 2)
        n_poses = len(w_joint)
        w_joint_sum = w_joint.sum(-1)
        idx_nonzero = w_joint_sum != 0.0
        w_joint[idx_nonzero] /= w_joint_sum[idx_nonzero][..., None]
        w_joint[~idx_nonzero] = 1.0 / s
        pose3Ds_joint_mean = pose3Ds * w_joint.transpose(1, 2)[..., None]
        pose3Ds_joint_mean = pose3Ds_joint_mean.sum(1)

        hm_joint_mean = eval_ptpose3Ds_on_hm(pose3Ds_joint_mean, idxs_op_to_simple, hmss, camera_matrixs)
        paf_joint_mean = eval_ptpose3Ds_on_paf(pose3Ds_joint_mean, idxs_op_to_simple, pafss, camera_matrixs)
        
        paf_joint_mean[paf_joint_mean < 0.0] = 0.0
        paf_joint_mean += eps
        paf_joint_mean *= (hm_joint_mean[:,IDX_PAF_JOINTS_FROM] + eps) * (hm_joint_mean[:,IDX_PAF_JOINTS_TO] + eps)
        w_result = (paf_joint_mean.norm(dim=-1) + eps).prod(-1).cpu().numpy()
        
        bools_survive = self.update_tracks(frame, ids, pose3Ds_joint_mean, np.log10(w_result))
        idx_survive = np.where(bools_survive)[0]
        ids = [ids[i] for i in idx_survive] 
        pose3Ds = pose3Ds[idx_survive]
        if len(pose3Ds_joint_mean) > 0:
            pose3Ds_joint_mean = pose3Ds_joint_mean[idx_survive]
        return ids, pose3Ds, pose3Ds_joint_mean.cpu().numpy()


    
    def update_tracks(self, frame, ids, pose3Ds, ws):
        """
        param frame : int
        param ids : [ int, ...]
        param pose3Ds : np.array { n x J x 3}
        param ws : [ float, ...]
        return bools_survive : [ bool, ...]
        """
        for (pose3D, id_pose, w) in zip(pose3Ds, ids, ws):
            if id_pose not in self.ids:
                self.ids.append(id_pose)
                self.frames.append(list())
                self.poses.append(list())
                self.ws.append(list())
                self.cnts_frames_triggered.append(0)
            idx = self.ids.index(id_pose)
            self.frames[idx].append(frame)
            self.poses[idx].append(pose3D.cpu().numpy())
            self.ws[idx].append(w)
            if w < self.thold_trigger:
                self.cnts_frames_triggered[idx] += 1
            else:
                self.cnts_frames_triggered[idx] = 0
        
        return [ self.ws[idx][-1] > self.thold_kill and self.cnts_frames_triggered[idx] <= self.n_frames_keep for idx in [self.ids.index(id_pose) for id_pose in ids] ]
        

    def refine(self):
        idx_invalid = []
        for i in range(len(self.ws)):
            idx_valid = np.where( np.array(self.ws[i]) > self.thold_trigger )[0]
            if len(idx_valid) > 0:
                idx_last = idx_valid[-1]
                self.frames[i] = self.frames[i][:(idx_last+1)]
                self.poses[i] = self.poses[i][:(idx_last+1)]
                self.ws[i] = self.ws[i][:(idx_last+1)]
            else:
                idx_invalid.append(i)

        if len(idx_invalid) > 0:
            self.ids = [self.ids[i] for i in range(len(self.ids)) if i not in idx_invalid]
            self.frames = [self.frames[i] for i in range(len(self.frames)) if i not in idx_invalid]
            self.poses = [self.poses[i] for i in range(len(self.poses)) if i not in idx_invalid]
            self.ws = [self.ws[i] for i in range(len(self.ws)) if i not in idx_invalid]
            self.cnts_frames_triggered = [self.cnts_frames_triggered[i] for i in range(len(self.cnts_frames_triggered)) if i not in idx_invalid]
    
    
    def get_poses_on_frame(self, frame):
        """
        param frame : int
        return ids
        return pose3Ds
        """
        ids = []
        pose3Ds = []
        for i in range(len(self.ws)):
            if frame in self.frames[i]:
                idx = self.frames[i].index(frame)
                ids.append(self.ids[i])
                pose3Ds.append(self.poses[i][idx])
        return ids, pose3Ds


    def save_tracks(self, fullfmt_out_track):
        # sig = 3
        # k = int((sig * 3)/2)*2 + 1
        # m = int(k/2)
        # gauss = np.exp(-(np.arange(k) - m)**2/(2*sig**2))
        # gauss = gauss/gauss.sum()
        # for i in range(len(self.ws)):
        #     poses = np.array(self.poses[i])
        #     l = len(poses)
        #     poses_pad = np.zeros((len(poses)+k-1,*poses.shape[1:]))
        #     poses_pad[m:m+l] = poses
        #     w_pad = np.pad(np.ones(l), (m, m), 'constant', constant_values=(0, 0))
            
        #     res = np.zeros((k,*poses.shape))
        #     w = np.zeros((k,l))
        #     for j in range(k):
        #         res[j]=poses_pad[j:l+j]
        #         w[j] = w_pad[j:l+j]
        #     w *= gauss[:,None]
        #     res = (res * w[...,None,None]).sum(0)/w[...,None,None].sum(0)
        #     self.poses[i] = res
        for i in range(len(self.ws)):
            with open(fullfmt_out_track.format(track = i), 'w') as f:
                json_track = {
                    "J": 18,
                    "z_axis": 2,
                    "frames": self.frames[i],
                    "poses": [trans_np_to_list((self.tf_matrixs_axis_scale @ pose3D.T).T,IDXS_SIMPL_TO_COCO) for pose3D in self.poses[i]],
                    "id": self.ids[i]
                }    
                json.dump(json_track, f)


    def load_tracks(self, fullfmt_out_track):
        self.ids = []
        self.frames = []
        self.poses = []
        self.ws = []
        self.cnts_frames_triggered = []
        i = 0
        while os.path.isfile(fullfmt_out_track.format(track = i)):
            print('load {}'.format(fullfmt_out_track.format(track = i)))
            with open(fullfmt_out_track.format(track = i)) as file:
                json_tracks = json.load(file)
                self.ids.append(json_tracks["id"])
                self.frames.append(json_tracks["frames"])
                self.poses.append([trans_list_to_np(listpose3D,IDXS_COCO_TO_SIMPL) for listpose3D in json_tracks["poses"]])
                assert len(self.frames[i]) == len(self.poses[i])
                self.ws.append([1.0] * len(self.poses))
                self.cnts_frames_triggered.append(0)
            i += 1
            