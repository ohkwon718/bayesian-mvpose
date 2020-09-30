import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from tqdm import tqdm

rainbow_cm = cm.gist_rainbow(np.swapaxes(np.linspace(0, 1, 100).reshape(10,10), 0, 1).flatten())
def save_imgseq_into_mp4(fullpath_out, fmt_img, frames, fps = 25):
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    capSize = cv2.imread(fmt_img.format(frame=frames[0]), cv2.IMREAD_COLOR).shape[1::-1]
    out = cv2.VideoWriter(fullpath_out, fourcc, fps, capSize)
    for frame in tqdm(frames):
        fname_img = fmt_img.format(frame=frame)
        img = cv2.imread(fname_img, cv2.IMREAD_COLOR)
        out.write(img)
    out.release()


def plot_nppose3Ds_projection(ax, pose3Ds, camera_matrix, connections, ids_pose = None, draw_points=True, draw_connections=True, kwargs_scatter = {}, kwargs_lines = {}):
    """
    :param ax: matplotlib subplot to plot onto
    :param pose3Ds: np.array{ n x j x 3 }
    :param camera_matrix: np.array{ 3 x 4 }
    :param connections: [ [idx, idx], ...] or np.array{ l x 2 }
    :param ids_pose: [ id, ... ] or np.array{ n }
    :param draw_points, draw_connections: bool
    :param kwargs_scatter, kwargs_lines: dict()
    """
    if len(pose3Ds) == 0:
        return

    assert isinstance(pose3Ds, np.ndarray)
    kwargs_scatter = kwargs_scatter.copy()
    kwargs_lines = kwargs_lines.copy()

    n, j, _ = pose3Ds.shape
    pose_xyz1 = np.ones((n,j,4))
    pose_xyz1[...,:3] = pose3Ds
    pose_xy1 = (camera_matrix @ pose_xyz1[...,None])[...,0]
    pose_xy1 /= pose_xy1[...,-1:]
    pose_xy = pose_xy1[...,:-1]

    if draw_connections:
        if not(isinstance(ids_pose, list) or isinstance(ids_pose, np.ndarray)) and ids_pose == None:
            ids_pose = np.arange(n)
        lines, colors_lines = [], []
        for connection in connections:
            lines.append(np.concatenate([pose_xy[:,connection[0],None],pose_xy[:,connection[1],None]], axis=1))
        lines = np.array(lines).reshape(-1, 2, 2)
        if 'colors' not in kwargs_lines:
            kwargs_lines['colors'] = np.tile(rainbow_cm[ids_pose % 25], (len(connections), 1))
        line_segments = LineCollection(lines, **kwargs_lines)
        ax.add_collection(line_segments)

    if draw_points:
        if not 'c' in kwargs_scatter:
            kwargs_scatter['c'] = rainbow_cm[np.tile(np.arange(j),n)]
        ax.scatter(x=pose_xy[...,0], y=pose_xy[...,1], **kwargs_scatter)

    ax.set_aspect('equal')


def plot_nppose3Ds(ax, pose3Ds, connections, ids_pose = None, draw_points=True, draw_connections=True, kwargs_scatter = {}, kwargs_lines = {}):
    """
    :param ax: matplotlib subplot to plot onto
    :param pose3Ds: np.array{ n x j x 3 }
    :param connections: [ [idx, idx], ...] or np.array{ l x 2 }
    :param ids_pose: [ id, ... ] or np.array{ n }
    :param draw_points, draw_connections: bool
    :param kwargs_scatter, kwargs_lines: dict()
    """
    if len(pose3Ds) == 0:
        return

    assert isinstance(pose3Ds, np.ndarray)
    kwargs_scatter = kwargs_scatter.copy()
    kwargs_lines = kwargs_lines.copy()
    
    n, j, _ = pose3Ds.shape
    if draw_connections:
        if not(isinstance(ids_pose, list) or isinstance(ids_pose, np.ndarray)) and ids_pose == None:
            ids_pose = np.arange(n)
        lines, colors_lines = [], []
        for connection in connections:
            lines.append(np.concatenate([pose3Ds[:,connection[0],None], pose3Ds[:,connection[1],None]], axis=1))
        lines = np.array(lines).reshape(-1, 2, 3)
        if not 'colors' in kwargs_lines:
            kwargs_lines['colors'] = np.tile(rainbow_cm[ids_pose % 25], (len(connections), 1))
        line_segments = Line3DCollection(lines, **kwargs_lines)
        ax.add_collection(line_segments)

    if draw_points:
        if not 'c' in kwargs_scatter:
            kwargs_scatter['c'] = rainbow_cm[np.tile(np.arange(j),n)]
        ax.scatter(pose3Ds[...,0], pose3Ds[...,1], pose3Ds[...,2], **kwargs_scatter)


