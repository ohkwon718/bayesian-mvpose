import numpy as np

JOINT_NAMES = ['Hip left', 'Knee left',  'Foot left',
                'Hip right', 'Knee right', 'Foot right',
                'Shoulder left', 'Elbow left', 'Hand left',
                'Shoulder right', 'Elbow right', 'Hand right', 
                'Neck', 'Nose' ]

CONNECTIONS = [ [0, 1], [1, 2], [0, 3],
                [3, 4], [4, 5],
                [0, 6], [6, 7], [7, 8],
                [3, 9], [9, 10], [10, 11],
                [9, 12], [12, 6],
                [12, 13] ]
                