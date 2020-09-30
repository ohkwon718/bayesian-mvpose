import numpy as np

cameras = [0,1,2,3,4]
fmt_calib = "../datasets/Shelf/calib/P{camera:}.txt"
CAMERA_MATRIXS_TO_IMG = np.array([np.loadtxt(fmt_calib.format(camera=camera), delimiter=",") for camera in cameras])

def get_calibration(dataset_name, cameras):
    return CAMERA_MATRIXS_TO_IMG[np.array(cameras)], None