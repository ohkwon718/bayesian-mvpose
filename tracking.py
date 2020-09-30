import os
import sys
import time
import shutil
import argparse
import json
import importlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import torch
from tqdm import tqdm

from view.openpose3d import get_pose3Ds_from_multi_hms
from view.stereo import get_fundamental_matrix_from_camera_matrix
from pose_models.pose_utils import get_trasforming_indices, trans_dict_to_flagged_np
import pose_models.openpose as pose_op
import pose_models.simplified as pose_simpl
from heatmap_subtraction.heatmap_subtraction import HeatmapSubtractor
from pose_tracking.pose_tracking import PoseTracking, eval_pose
from pose_tracking.postprocessing import Tracker
from utils.visualize import save_imgseq_into_mp4, plot_nppose3Ds_projection, plot_nppose3Ds


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

fmt_out_track = "track/track{track:}.json"
fmt_out_plot = "plot_process/plot-{frame:05}.png"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

parser = argparse.ArgumentParser(description='Recursive Bayesian Filtering for MultipleHuman Pose Tracking from Multiple Cameras')
parser.add_argument('--json-dataset', type=str, default='', help='json file to load datsaset settings')
parser.add_argument('--json-heatmap-substract', type=str, default='./heatmap_subtraction/setup.json', help='json file to setup cnn heatmap-substract model')
parser.add_argument('--json-pose-prediction', type=str, default='./pose_prediction/setup.json', help='json file to setup pose-prediction model')
parser.add_argument('--frame-start', type=int, default=-1, help='start frame')
parser.add_argument('--frame-end', type=int, default=-1, help='end frame')
parser.add_argument('--path-out', type=str, default='../results/mvpose3d_tracking/', help='output path')
parser.add_argument('--visualize', action='store_true', default=False, help='visualize the result')
parser.add_argument('--num-sample-pose', type=int, default=600, help='number of sample hypothesis for each pose')
parser.add_argument('--p-completion', type=float, default=0.7, help='dropout rate for bayesian pose completion')
parser.add_argument('--p-prediction', type=float, default=0.8, help='dropout rate for bayesian pose prediction')
parser.add_argument('--eps', type=float, default=0.1, help='tolerance parameter for pose evaluation')
parser.add_argument('--num-random-search', type=int, default=100, help='number of sample for random search')
parser.add_argument('--std-joint', type=float, default=0.03, help='dropout rate for bayesian pose prediction')
parser.add_argument('--thold-kill', type=float, default=-9, help='threshold to kill a track completely')
parser.add_argument('--thold-trigger', type=float, default=-7, help='threshold to trigger disappearing')
parser.add_argument('--n-frames-keep', type=float, default=10, help='frame numbers in disappearing before killing a track completely')
parser.add_argument('--thold-hm-peak', type=float, default=0.3, help='threshold to detect peaks on heatmaps')
parser.add_argument('--thold-sv', type=float, default=15.0, help='threshold for singular values on stereo matching')
parser.add_argument('--thold-dist-joint', type=float, default=0.05, help='distance between joint candidates to be merged')
parser.add_argument('--thold-paf', type=float, default=0.7, help='paf threshold to determine combining poses and joints')
parser.add_argument('--least-joint-num', type=int, default=5, help='least number of joints to build a pose')
parser.add_argument('--thold-dist-pose', type=float, default=0.3, help='distance between poses to be clustered')
parser.add_argument('--num-sample-paf', type=int, default=10, help='number of samples which are used for integration of part affinity fields')
args = parser.parse_args()

np.set_printoptions(suppress=True)
assert args.json_dataset != ''
with open(args.json_dataset) as json_file:
    settings_dataset = json.load(json_file)
dataset_name = frame_start = frame_end = fmt_img = fmt_hp = import_calib = cameras = None
fig_size = fig_grid = subplot_cam = subplot_3d = plt3d_lim = None
scale = 1.0
axis = [[0,1,2], [1,1,1]]
locals().update(settings_dataset)
axis = np.array(axis)

if args.frame_start != -1:
    frame_start = args.frame_start
if args.frame_end != -1:
    frame_end = args.frame_end
frames = range(frame_start, frame_end+1)

n_camera = len(cameras)
n_camera_pairs = int((n_camera * (n_camera-1))/2)
n_sample_pose = int(args.num_sample_pose / n_camera_pairs) * n_camera_pairs

path_out = os.path.join( args.path_out, dataset_name )
print(path_out)
fullfmt_out_track = os.path.join( path_out, fmt_out_track)
fullfmt_out_plot = os.path.join( path_out, fmt_out_plot)
fullpath_out_mp4 = os.path.join( path_out, "{}_tracking.mp4".format(dataset_name))

if os.path.isdir(path_out):
    while True:
        print("The output folder already exists. Do you want to delete it and keep running?[y/n] ".format(path_out), end='')
        answer = input().lower()
        if answer == "y":
            shutil.rmtree(path_out)
            break
        elif answer == "n":
            sys.exit()
    
os.makedirs(os.path.dirname(fullfmt_out_track))
if args.visualize:
    os.makedirs(os.path.dirname(fullfmt_out_plot))

SIZE_IMG = cv2.imread(fmt_img.format(frame=frame_start, camera=cameras[0]), cv2.IMREAD_COLOR).shape[:2]
h_hp, w_hp = cv2.imread(fmt_hp.format(frame=frame_start, camera=cameras[0]), cv2.IMREAD_GRAYSCALE).shape
SIZE_HM = h_hp, int(w_hp/78)
calib = importlib.import_module(import_calib)
CAMERA_MATRIXS_TO_IMG, CAMERA_PARAMS_UNDIST = calib.get_calibration(dataset_name, cameras)
TF_MATRIXS_AXIS_SCALE = np.diag(np.append(axis[-1], 1.0/scale))[np.append(axis[0],-1)]
CAMERA_MATRIXS_TO_IMG = CAMERA_MATRIXS_TO_IMG @ TF_MATRIXS_AXIS_SCALE
CAMERA_MATRIXS_TO_HP = np.array([np.diag(np.array(SIZE_HM + (1,))/np.array(SIZE_IMG + (1,))) @ camera_matrixs for camera_matrixs in CAMERA_MATRIXS_TO_IMG])
CAMERA_MATRIXS_TO_HP_TORCH = torch.from_numpy(CAMERA_MATRIXS_TO_HP).float().to(device)
FUNDAMENTAL_MATRIXS_TO_HP = get_fundamental_matrix_from_camera_matrix(CAMERA_MATRIXS_TO_HP)
IDXS_OP_TO_SIMPL = get_trasforming_indices(pose_op.JOINT_NAMES, pose_simpl.JOINT_NAMES)

assert args.json_heatmap_substract != ''
print("Load Heatmap Substraction Model...")
hs = HeatmapSubtractor(args.json_heatmap_substract, device = device)

assert args.json_pose_prediction != ''
print("Load Pose Prediction Model...")  
pp = PoseTracking(args.json_pose_prediction, n_camera = n_camera, n_sample = n_sample_pose, device = device, n_random_search = args.num_random_search)

torch.set_grad_enabled(False)

plt.ioff()
fig = plt.figure(figsize=fig_size)
gs = gridspec.GridSpec(*fig_grid, figure = fig)
kwargs_prediction = { 'draw_connections' : False, 
    'draw_points' : True, 'kwargs_scatter' : {'s':1.0, 'alpha':1.0, 'edgecolors':'black', 'linewidths':0.5} }
kwargs_update = { 'draw_connections' : True, 'kwargs_lines' : {'linewidth':1, 'alpha':0.01}, 
                'draw_points' : True, 'kwargs_scatter' : {'s':1.5, 'alpha':0.01} }
kwargs_result = { 'draw_connections' : True, 'kwargs_lines' : {'linewidth':1, 'alpha':0.8, 'color':'black'}, 
                'draw_points' : False}

batch_simpl3Ds = torch.zeros(0, n_sample_pose, 14, 3).to(device)
ids_track = []
id_new = 0
tracks = Tracker(args.thold_trigger, args.thold_kill, args.n_frames_keep, CAMERA_MATRIXS_TO_HP, scale)

time_elapsed = 0.0
print("Pose Tracking")
loop_frames = tqdm(frames)
for frame in loop_frames:
    time_start = time.time()
    loop_frames.set_description("Frame {}, {} tracks, {:3.2f}sec/frame".format(frame, len(ids_track), time_elapsed))
    hps_reshape = []
    for i_c, camera in enumerate(cameras):
        fname_hp = fmt_hp.format(frame=frame, camera=camera)        
        img_raw = cv2.imread(fname_hp, cv2.IMREAD_GRAYSCALE)
        h_raw, w_raw = img_raw.shape
        hp_reshape = np.transpose(img_raw.reshape((h_raw,78,int(w_raw/78))), (1, 0, 2))
        if CAMERA_PARAMS_UNDIST:
            K, distCoef, K_undist = CAMERA_PARAMS_UNDIST[i_c]
            K_hp = np.diag(np.array(SIZE_HM + (1,))/np.array(SIZE_IMG + (1,))) @ K
            K_hp_undist = np.diag(np.array(SIZE_HM + (1,))/np.array(SIZE_IMG + (1,))) @ K_undist
            hp_reshape = np.transpose(cv2.undistort(np.transpose(hp_reshape, (1,2,0)), K_hp, distCoef, None, K_hp_undist),(2,0,1))
        hps_reshape.append(hp_reshape)
    hps_reshape = np.array(hps_reshape)
    hmss = hps_reshape[:,:25].astype(float) / 255.
    pafss = (hps_reshape[:,26:].astype(float) - 128.)/128.

    hmss_torch = hmss.copy()
    hmss_torch[:,0] = np.max(hmss_torch[:,(0,-4,-3,-2,-1)], 1)
    hmss_torch = torch.from_numpy(hmss_torch).float().to(device)
    pafss_torch = torch.from_numpy(pafss).float().to(device)
    ##################################################################################################################
    time_start = time.time()
    batch_simpl3Ds = pp.prediction(batch_simpl3Ds, ids = ids_track, p = args.p_prediction, dropout = True)
    nppose3Ds_prediction = batch_simpl3Ds.cpu().numpy().reshape(-1, 14, 3) ##
    ids_prediction = np.repeat(np.array(ids_track), n_sample_pose) ##
    batch_simpl3Ds = pp.update(batch_simpl3Ds, ids_track, IDXS_OP_TO_SIMPL, hmss_torch, pafss_torch, CAMERA_MATRIXS_TO_HP_TORCH, std_joint = args.std_joint, eps = args.eps)    
    time_end = time.time()
    time_elapsed = time_end - time_start
    ids_track, batch_simpl3Ds, simpl3Ds_result = tracks.update(frame, ids_track, batch_simpl3Ds, IDXS_OP_TO_SIMPL, hmss_torch, pafss_torch, CAMERA_MATRIXS_TO_HP_TORCH)
    
    batch_simpl3Ds_newcomer = torch.zeros(0, n_sample_pose, 14, 3).to(device)
    n_pose_temporal_prev = -1
    while True:
        hmss_newcomer = hs.get_newcomer(torch.cat((batch_simpl3Ds, batch_simpl3Ds_newcomer)), IDXS_OP_TO_SIMPL, hmss, CAMERA_MATRIXS_TO_HP_TORCH)
        hmss_newcomer = hmss_newcomer.cpu().numpy()
        openpose3Ds_temporal, joint3d_candidates, joint2d_candidates = get_pose3Ds_from_multi_hms(hmss_newcomer, pafss, CAMERA_MATRIXS_TO_HP, FUNDAMENTAL_MATRIXS_TO_HP,
                        thold_paf = args.thold_paf,
                        min_num_of_joint = args.least_joint_num, thold_cluster = args.thold_dist_joint,
                        thold_hm_peak = args.thold_hm_peak, thold_sv = args.thold_sv, 
                        num_sample_paf = args.num_sample_paf)
        openpose3Ds_temporal = [{key:openpose3D[key] for key in openpose3D if key in IDXS_OP_TO_SIMPL} for openpose3D in openpose3Ds_temporal]
        openpose3Ds_temporal = [openpose3D for openpose3D in openpose3Ds_temporal if len(openpose3D) >= args.least_joint_num]
        n_pose_temporal = len(openpose3Ds_temporal)
        if n_pose_temporal == 0 or n_pose_temporal == n_pose_temporal_prev:
            break
        n_pose_temporal_prev = n_pose_temporal

        fsimpl3Ds_temporal = np.array([trans_dict_to_flagged_np(openpose3D, IDXS_OP_TO_SIMPL) for openpose3D in openpose3Ds_temporal])
        clusters_fsimpl3Ds_temporal = [[ fsimpl3D ] for fsimpl3D in fsimpl3Ds_temporal]
        
        batch_simpl3Ds_complete = pp.completion(clusters_fsimpl3Ds_temporal, p = args.p_completion, dropout = True)
        w_pose3Ds = eval_pose(batch_simpl3Ds_complete.cpu().numpy(), IDXS_OP_TO_SIMPL, hmss_newcomer, pafss, CAMERA_MATRIXS_TO_HP)
        
        idx_best = np.unravel_index(w_pose3Ds.argmax(), w_pose3Ds.shape)[0]
        batch_simpl3Ds_best = batch_simpl3Ds_complete[idx_best, None]
        batch_simpl3Ds_best = pp.update(batch_simpl3Ds_best, None, IDXS_OP_TO_SIMPL, hmss_torch, pafss_torch, CAMERA_MATRIXS_TO_HP_TORCH, std_joint = args.std_joint, eps = args.eps)
        batch_simpl3Ds_newcomer = torch.cat((batch_simpl3Ds_newcomer, batch_simpl3Ds_best))
        

    if len(batch_simpl3Ds_newcomer) > 0:
        n_newcomer = len(batch_simpl3Ds_newcomer)
        ids_track_newcomer = list(range(id_new, id_new + n_newcomer))
        ids_track_newcomer, batch_simpl3Ds_newcomer, simpl3Ds_result_newcomer = tracks.update(frame, ids_track_newcomer, batch_simpl3Ds_newcomer, IDXS_OP_TO_SIMPL, hmss_torch, pafss_torch, CAMERA_MATRIXS_TO_HP_TORCH)
        
        batch_simpl3Ds = torch.cat((batch_simpl3Ds, batch_simpl3Ds_newcomer))
        ids_track = ids_track + ids_track_newcomer
        id_new += n_newcomer
        
    ##################################################################################################################
    
    if args.visualize:
        ids_update = np.repeat(np.array(ids_track), n_sample_pose)
        nppose3Ds_update = batch_simpl3Ds.cpu().numpy().reshape(-1,14,3)
        ids_result = np.repeat(np.array(ids_track), n_sample_pose)
        nppose3Ds_result = simpl3Ds_result

        fig.clf()
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.01, hspace=0.01)
        for i_c, camera in enumerate(cameras):
            gs_idx, gs_span = subplot_cam[i_c]
            ax = fig.add_subplot(gs[ gs_idx[0]:(gs_idx[0]+gs_span[0]), gs_idx[1]:(gs_idx[1]+gs_span[1]) ])
            ax.axis('off')
            fname_img = fmt_img.format(frame=frame, camera=camera)
            img = cv2.imread(fname_img, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if CAMERA_PARAMS_UNDIST:
                K, distCoef, K_undist = CAMERA_PARAMS_UNDIST[i_c]
                img = cv2.undistort(img, K, distCoef, None, K_undist)
            ax.imshow(img)
            
            plot_nppose3Ds_projection(ax, nppose3Ds_prediction, CAMERA_MATRIXS_TO_IMG[i_c], pose_simpl.CONNECTIONS, ids_pose = ids_prediction, **kwargs_prediction)
            plot_nppose3Ds_projection(ax, nppose3Ds_update, CAMERA_MATRIXS_TO_IMG[i_c], pose_simpl.CONNECTIONS, ids_pose = ids_update, **kwargs_update)
            plot_nppose3Ds_projection(ax, nppose3Ds_result, CAMERA_MATRIXS_TO_IMG[i_c], pose_simpl.CONNECTIONS, ids_pose = ids_result, **kwargs_result)
            ax.set_xlim([0,img.shape[1]]) 
            ax.set_ylim([img.shape[0],0])
            
        gs_idx, gs_span = subplot_3d
        ax = fig.add_subplot(gs[ gs_idx[0]:(gs_idx[0]+gs_span[0]), gs_idx[1]:(gs_idx[1]+gs_span[1]) ], projection='3d')
        ax.set_xlim(*plt3d_lim[0])
        ax.set_ylim(*plt3d_lim[1])
        ax.set_zlim(*plt3d_lim[2])
        plot_nppose3Ds(ax, nppose3Ds_prediction, pose_simpl.CONNECTIONS, ids_pose = ids_prediction, **kwargs_prediction)
        plot_nppose3Ds(ax, nppose3Ds_update, pose_simpl.CONNECTIONS, ids_pose = ids_update, **kwargs_update)
        plot_nppose3Ds(ax, nppose3Ds_result, pose_simpl.CONNECTIONS, ids_pose = ids_result, **kwargs_result)
        
        fig.savefig(fullfmt_out_plot.format(frame=frame), bbox_inches='tight', pad_inches=0)


tracks.save_tracks(fullfmt_out_track)

if args.visualize:
    print("converting into {}...".format(fullpath_out_mp4))
    save_imgseq_into_mp4(fullpath_out_mp4, fullfmt_out_plot, frames)

print("Done at {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())))
