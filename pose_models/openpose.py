import numpy as np

# remarks : official openpose includes more parts
JOINT_NAMES = ['Nose', 'Neck', 
                'Shoulder right', 'Elbow right', 'Hand right', 
                'Shoulder left', 'Elbow left', 'Hand left',
                'Hip center',
                'Hip right', 'Knee right', 'Foot right',
                'Hip left', 'Knee left',  'Foot left',
                'Eye right', 'Eye left', 'Ear right', 'Ear left']
    # 'toe1 left',
    # 'toe2 left',
    # 'heel left',
    # 'toe1 right',
    # 'toe2 right',
    # 'heel right',


CONNECTIONS = [[1,8],[9,10],[10,11],[8,9],[8,12],[12,13],[13,14],[1,2],
                        [2,3],[3,4],[1,5],[5,6],[6,7],[1,0],
                        [2,17],[5,18],[0,15],[0,16],[15,17],[16,18],[17,18]]
