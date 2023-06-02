# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 12:37:31 2022

@author: EmmaWoods
"""
import json

#%%


# AXIS_RANGE = { "x-axis": [ -65, 165 ],
#                "y-axis": [ -65,  65 ],
#                "z-axis": [   0, 235 ],
#                "a-axis": [   0,  45 ],
#                "b-axis": [   0, 360 ] }    

# Range for: x, y, z, a, b 
AXIS_RANGE = {"min": [-65,  -65,    0,   0,   0],
              "max": [165,   65,  235,  45, 340]}

# to be updated with motion characterisation script
CAMERA_CALIBRATION_AXIS_RANGE = {"min": [-65,  -65,    0,   0,   0],
                                 "max": [165,   65,  235,  45, 340]}

CAMERA_OFFSETS= {
        0: 25,
        1: -10,
        "system": 8.36 
        }

# nominal location that should be able to see the centre
NOMINAL_LOC = [0, 0, 90, 30, 0], 
# Offset from camera to centre of stage 
XYZ_OFFSET = [357, -1, 175],
# Camera working distance
WORKING_DIST = 435
# Camera names for save files
CAMERA_NAMES = ["camera_00","camera_01"]


system_data = {
    "axis_range": AXIS_RANGE, 
    "calibration_axis_range": CAMERA_CALIBRATION_AXIS_RANGE,
    "camera_offsets": CAMERA_OFFSETS, 
    "nominal_loc": NOMINAL_LOC,
    "xyz_offset": XYZ_OFFSET,
    "working_distance": WORKING_DIST,
    "camera_names": CAMERA_NAMES
    }

with open('sys_config.json', 'w') as fp:
    json.dump(system_data, fp, indent=4)