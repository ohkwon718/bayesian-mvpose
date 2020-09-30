import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
def get_joint_indices(joint_names):
    ret = dict()
    for i, joint_name in enumerate(joint_names):
        ret[joint_name] = i
    return ret

def get_trasforming_indices(joint_names_from, joint_names_to):
    idxs = [None] * len(joint_names_to)
    for i_to, joint_name_to in enumerate(joint_names_to):
        for i_from, joint_name_from in enumerate(joint_names_from):
            if joint_name_to == joint_name_from:
                idxs[i_to] = i_from
    return idxs

def trans_dict_to_dict(pose, indice_transform):
    """
    param pose : dict()
    param indice_transform : [int, int, int, ...]
    return : dict()
    """
    ret = {}
    for idx_ret, idx in enumerate(indice_transform):
        if idx in pose.keys():
            ret[idx_ret] = pose[idx]
    return ret

def trans_dict_to_flagged_np(pose, indice_transform):
    """
    param pose : dict()
    param indice_transform : [int, int, int, ...]
    return : np.array{ j x 4 }
    """
    ret = np.zeros((len(indice_transform),4))
    for idx_ret, idx in enumerate(indice_transform):
        if idx in pose.keys():
            ret[idx_ret, :3] = pose[idx]
            ret[idx_ret, -1] = 1
    return ret


def trans_flagged_np_to_dict(pose, indice_transform):
    """
    param pose : np.array{ j x 4 }
    param indice_transform : [int, int, int, ...]
    return : dict()
    """
    ret = {}
    for i, joint in enumerate(pose):
        if joint[-1] == 1.0:
            ret[indice_transform[i]]= joint[:-1]
    return ret

def trans_dict_to_np(pose, indice_transform):
    """
    param pose : dict()
    param indice_transform : [int, int, int, ...]
    return : np.array{ j x 3 }
    """
    ret = np.zeros((len(indice_transform),3))
    for idx_ret, idx in enumerate(indice_transform):
        if idx in pose.keys():
            ret[idx_ret] = pose[idx]
    return ret

def trans_np_to_dict(pose, indice_transform):
    """
    param pose : np.array{ j x 3 }
    param indice_transform : [int, int, int, ...]
    return : dict()
    """
    ret = {}
    for i, joint in enumerate(pose):
        ret[indice_transform[i]]= joint
    return ret

def trans_np_to_list(pose, indice_transform):
    """
    param pose : np.array{ j x 3 }
    param indice_transform : [int, int, int, ...]
    return : dict()
    """
    ret = []
    for idx in indice_transform:
        if idx == None:
            ret.append(None)
        else:
            ret.append(pose[idx].tolist())
    return ret

def trans_list_to_np(pose, indice_transform):
    """
    param pose : dict()
    param indice_transform : [int, int, int, ...]
    return : np.array{ j x 3 }
    """
    ret = np.zeros((len(indice_transform),3))
    for idx_ret, idx in enumerate(indice_transform):
        if idx < len(pose):
            ret[idx_ret] = pose[idx]
    return ret

def dist_btw_dict(pose1, pose2):
    """
    param pose : dict()
    return : float
    """
    shared_joints = list(set(pose1) & set(pose2))
    if shared_joints:
        dists = []
        for joint in shared_joints:
            dists.append(np.linalg.norm(pose1[joint] - pose2[joint]))
        dist = np.array(dists).mean(0)
    else:
        dist = np.linalg.norm(np.array(list(pose1.values())).mean(0) - np.array(list(pose2.values())).mean(0))
    return dist

def hcluster_on_dicts(poses, threshold_cluster = 0.1):
    """
    param poses : [ dict(), ... ]
    return : [ [int, int, ...], ... ] which are indices of poses
    """
    if len(poses) == 0:
        return []
    if len(poses) == 1:
        return [[0]]
    dist_matrix = np.array([[dist_btw_dict(pose1, pose2) for pose2 in poses] for pose1 in poses])
    dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
    Z = linkage(squareform(dist_matrix), 'ward')
    cluster = fcluster(Z, t=threshold_cluster, criterion='distance')
    clusters_idxs = []
    for c in range(cluster.max()):
        clusters_idxs.append([i for i, isCluster in enumerate(cluster == c+1) if isCluster])
    return clusters_idxs

def hcluster_on_np(poses, threshold_cluster = 0.1):
    """
    param poses : np.array { n x j x 3} or np.array { n x j x 2}
    return : [ [int, int, ...], ... ] which are indices of poses
    """
    if len(poses) == 0:
        return []
    if len(poses) == 1:
        return [[0]]
    dist_matrix = np.array([[np.linalg.norm(pose1 - pose2, axis = -1).mean() for pose2 in poses] for pose1 in poses])
    dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
    Z = linkage(squareform(dist_matrix), 'ward')
    cluster = fcluster(Z, t=threshold_cluster, criterion='distance')
    clusters_idxs = []
    for c in range(cluster.max()):
        clusters_idxs.append([i for i, isCluster in enumerate(cluster == c+1) if isCluster])
    return clusters_idxs