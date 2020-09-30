import numpy as np
import itertools
from operator import itemgetter
from scipy.linalg import lstsq
from view.camera import is_front_of_view


def get_cross(A):
    return np.array([[0, -A[2], A[1]], [A[2], 0, -A[0]], [-A[1], A[0], 0]])


def get_fundamental_matrix_from_camera_matrix(camera_matrixs):
    fundamental_matrixs = {}
    camera_centers = [-np.linalg.inv(camera_matrix[:,:3]) @ camera_matrix[:,3] for camera_matrix in camera_matrixs]
    for i in range(len(camera_centers)):
        for j in range(len(camera_centers)):
            if i == j:
                continue
            e = camera_matrixs[j] @ np.append(camera_centers[i],1)
            e = e/e[2] 
            fundamental_matrixs[i,j] = get_cross(e) @ camera_matrixs[j] @ np.linalg.pinv(camera_matrixs[i])
    return fundamental_matrixs


def get_triangulation(xys, camera_matrixs):
    """
    param xys : list of points in cameras
    param camera_matrixs : np.array{ v x 3 x 4 }
    return X,s : the point in 3d space and corresponding singular value
    """
    A = np.empty((0,4))
    for x, camera_matrix in zip(xys, camera_matrixs):
        A = np.append(A, x[:2].reshape(2,1) @ camera_matrix[2:3] - camera_matrix[:2], axis = 0)
    
    _,s,vh = np.linalg.svd(A)
    X = vh[-1]
    X = X/X[-1]
    
    return X[:-1], s[-1]


def test_epipolar_constraints(xys, fundamental_matrixs, idx, thr_pixel = 50):
    for i in range(len(xys)):
        for j in range(len(xys)):
            if i == j:
                continue
            epi_norm = np.append(xys[i],1) @ fundamental_matrixs[j,i]
            epi_norm /= np.linalg.norm(epi_norm[:2])
            pt_err = epi_norm @ np.append(xys[j],1)
            if pt_err > thr_pixel:
                return False
    return True

def get_3d_points_from_multiple_observation(candidate_sets, camera_matrixs, fundamental_matrixs, threshold_sv = 15.0):
    """
    param candidate_sets : [[np.array{2}, ...], ...] list of candidate sets(list) in cameras
    param camera_matrixs : np.array{ v x 3 x 4 }
    return : [np.array{3}, ...] the corresponding points in 3d space
    """
    for candidate_set in candidate_sets:
        candidate_set.append(None)
    combinations = list(itertools.product(*candidate_sets))
    combinations = [(combination, len([combination[i] for i in range(len(combination)) if combination[i] != None])) for combination in combinations]
    combinations = sorted(combinations, key=itemgetter(1), reverse = True)
    combinations = [combination[0] for combination in combinations]
    
    combinations_used = []
    res = []

    for combination in combinations:
        combination_valid = [combination[i] for i in range(len(combination)) if combination[i] != None]
        if len(combination_valid) >= 2:
            isSubset = False
            for combination_used in combinations_used: #if not subset of previous
                isSubset = True
                for pt_tgt, pt_used in zip(combination, combination_used):
                    if pt_tgt != pt_used and not pt_tgt == None:
                        isSubset = False
                        break
                if isSubset:
                    break
            # if not isSubset:
            #     idx_valid = np.array([i for i in range(len(combination)) if combination[i] != None])
            #     combination_valid = [combination[i] for i in idx_valid]
            #     camera_matrixs_valid = camera_matrixs[idx_valid]
            #     if test_epipolar_constraints(np.vstack(combination_valid), fundamental_matrixs, idx_valid):
            #         xyz, s = get_triangulation(np.vstack(combination_valid), camera_matrixs_valid)
            #         s /= len(camera_matrixs_valid)
            #         if s < threshold_sv and np.all(is_front_of_view(xyz, camera_matrixs_valid)):
            #             combinations_used.append(combination)
            #             res.append(xyz)
            if not isSubset:
                camera_matrixs_valid = np.array([camera_matrixs[i] for i in range(len(combination)) if combination[i] != None])
                xyz, s = get_triangulation(np.vstack(combination_valid), camera_matrixs_valid)
                s /= len(camera_matrixs_valid) # to reward points with many views
                if s < threshold_sv and np.all(is_front_of_view(xyz, camera_matrixs_valid)):
                    combinations_used.append(combination)
                    res.append(xyz)

    return res


