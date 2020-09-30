import numpy as np

cameras = [0,1,2]
fmt_calib = "../datasets/campus/calib/P{camera:}.txt"
CAMERA_MATRIXS_TO_IMG = np.array([np.array([[3.,0.,0.],[0.,3.,0.],[0.,0.,1.]]) @ np.loadtxt(fmt_calib.format(camera=camera), delimiter=" ") for camera in cameras])


def get_calibration(dataset_name, cameras):
    return CAMERA_MATRIXS_TO_IMG, None