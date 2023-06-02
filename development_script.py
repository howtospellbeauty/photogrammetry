# -*- coding: utf-8 -*-
"""
Created on Tue May 4 2022

@author: Wenwen

File for current development of a module. 

CURRENTLY WORKING ON: CAMERA CHARACTERISATION
DATE STARTED: 27/4/23

TO DO: 
    1) Find axis range in which images ret checkerboard for each cam/both
    2) Find bias ratio to find out camera model:
          i) find intrinsics using dataset
         ii) deconsctruct image to make dataset of squares
        iii) reestimate pose using each new image
         iV) compute MSE over all residuals
    3) Uncertainty analysis

"""
import os
import sys
sys.path.insert(1,'./source/')
sys.path.insert(1,'./data/')

import json
import fnmatch
import cv2 as cv
import numpy as np
from PIL import Image
import scipy.stats as sts
import matplotlib.pyplot as plt

import system_control as sc



# Starting on objective 2

def detect_checkerboard(img, grid, spacing,resize_ratio):
    
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS +
                cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)        

    
    # # resize image to speed up detection
    # new_image_size = (int((img.shape[1])/resize_ratio),
    #                   int((img.shape[0])/resize_ratio))
        
    # smaller_image = cv.resize(img.astype("uint8"),
    #                           new_image_size,
    #                           interpolation=cv.INTER_AREA)
    
    # find chessboard corners for smaller image
    ret, corners = cv.findChessboardCorners(img,
                                           (grid[0],grid[1]),
                                           None)
    
    
    # if found do refined corner detection
    if ret:
        # Refined corner detection
        corners_sub = cv.cornerSubPix(img.astype("uint8"),
                                      corners,
                                      (11, 11),
                                      (-1, -1),
                                      criteria)
        return ret, corners_sub
    else:
        return ret, []


# Checkerboard properties
# Grid size
grid = [11,12]
grid = [17,24]
# grid = [7, 10]

# number of non-corner sharing black squares in grid
no_squares = int(((grid[0]-1)/2)*(grid[1]/2))
no_corners_black = no_squares*4
no_corners_all = grid[0]*grid[1]

# Grid spacing
spacing = 6
spacing = 7.5
# spacing = 20.5


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((grid[0]*grid[1],3), np.float32)
objp[:,:2] = np.mgrid[0:grid[0],0:grid[1]].T.reshape(-1,2)
objp = objp * spacing

# resize ratio to reduce image size
resize_ratio = 2**3    

directory = "data/calibration_image_dataset/"


# pre-allocate array of images per camera
images_00 = []

# check image directory exists    
if os.path.exists(directory) == False:
    raise ValueError("No calibration file directory - take new images")

# number of calibration images
no_images = len(fnmatch.filter(os.listdir(directory), '*.tiff'))  
image_names = os.listdir(directory)


for n in range(no_images):
    
    current_image = cv.imread(directory +image_names[n])
    images_00.append(cv.cvtColor(current_image,cv.COLOR_BGR2GRAY)) 




ret_images_00 = []
# Arrays to store object and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints_00 = [] # 2d points in image plane


for m in range(len(images_00)):
    
    ret00, corners00 = detect_checkerboard(images_00[m],
                                           grid, 
                                           spacing, 
                                           resize_ratio)

    
    if ret00:
        ret_images_00.append(images_00[m])
        objpoints.append(objp)
        imgpoints_00.append(corners00)
        # cv.drawChessboardCorners(images_00[m], grid, corners00, ret00)
        # imS = cv.resize(images_00[m], (1920, 1080)) 
        # cv.imshow('test',imS) 

        
print("no images cam 00: "+str(len(ret_images_00)))

# Initial Calibration To Extract Intrinsics

cal_flag = (cv.CALIB_FIX_ASPECT_RATIO   +
            cv.CALIB_RATIONAL_MODEL +
            # cv.CALIB_ZERO_TANGENT_DIST +  
            cv.CALIB_FIX_K1 +
            cv.CALIB_FIX_K2 +
            cv.CALIB_FIX_K3 +
            cv.CALIB_FIX_K4 +
            cv.CALIB_FIX_K5 +
            cv.CALIB_FIX_K6
            )
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS +
            cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)  
    
ret_00, mtx_00, dist_00, rvecs_00, tvecs_00 = cv.calibrateCamera(
                                        objpoints, 
                                        imgpoints_00, 
                                        images_00[0].shape[::-1],
                                        None,
                                        None,
                                        flags=cal_flag, 
                                        )







## Trying to implement calculation of bias ratio from Hagemann paper. 

plt.close('all')


no_images = len(ret_images_00)
no_extrinsics = 6
no_corners_total = 2*no_corners_all*no_images
 
for j in range(no_images):   
    # select current image 
    im = ret_images_00[j]
    ret, corners = detect_checkerboard(im, grid, spacing, resize_ratio)
    if ret:
        # #show all corners
        # plt.figure()
        # plt.imshow(im)
        # plt.plot(corners[:,0,0], corners[:,0,1], 'rx')
        # plt.show()
        
        
        # loop through individual squares in current image
        corner1 = 0 # initial index value for corner 1
        unused = 4  # after this square, next corner index not used. Gets updated later
        rep_er = [] # pre-allocate array for reprojection errors
        MSE_v = [] # pre-allocate array for virtual target MSE
        
        for i in range(no_squares):
            
            # assign other corner indices
            corner2 = corner1 + 1
            corner3 = corner1 + 11
            corner4 = corner1 + 12
            
            # # Segment Image to Find Pose
            # plt.figure()
            # plt.imshow(im)
            # plt.plot(corners[corner1:corner2+1,0,0], 
            #           corners[corner1:corner2+1,0,1], 
            #           'rx')
            # plt.plot(corners[corner3:corner4+1,0,0], 
            #           corners[corner3:corner4+1,0,1], 
            #           'rx')
            # plt.show()
            
            
            corners_current = np.float32(np.array(
                [[corners[corner1,0,0], corners[corner1,0,1], 0], 
                 [corners[corner2,0,0], corners[corner2,0,1], 0], 
                 [corners[corner3,0,0], corners[corner3,0,1], 0], 
                 [corners[corner4,0,0], corners[corner4,0,1], 0]]))
            
            # ----------------------------------------------------------------------------
            # # TODO: don't neccessarily need this bit of code was useful for 
            # # visualisation purposes
            # # number of pixels as a boarder around the square of interest
            # widen_factor = 50
            
            # start_point = (int(np.min(corners_current[:,0])-widen_factor), 
            #                int(np.min(corners_current[:,1])-widen_factor))
            # end_point = (int(np.max(corners_current[:,0])+widen_factor), 
            #              int(np.max(corners_current[:,1])+widen_factor))
            
            # color = 255
            
            # plt.figure()
            # mask = np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8)
            # cv.rectangle(mask,start_point, end_point, color ,-1)
            # plt.plot(corners[corner1:corner2+1,0,0], 
            #           corners[corner1:corner2+1,0,1], 
            #           'rx')
            # plt.plot(corners[corner3:corner4+1,0,0], 
            #           corners[corner3:corner4+1,0,1], 
            #           'rx')
            # plt.imshow(mask)
            
            # plt.figure()
            # masked = cv.bitwise_and(im, im, mask=mask)
            # plt.imshow(masked)
            
            # ----------------------------------------------------------------------------
            
            # object points for current square of interest
            objp_current = np.vstack([[objp[corner1]], 
                                      [objp[corner2]], 
                                      [objp[corner3]], 
                                      [objp[corner4]]])
            
            # reshape corners array to use in PnP
            corners_current = np.ascontiguousarray(
                corners_current[:,:2]).reshape((4,1,2))
            
            # Solve for only 4 points
            flags = cv.SOLVEPNP_P3P
            
            # Reestimate the pose of the square of interest
            ret, rvecs, tvecs = cv.solvePnP(objp_current,
                                            corners_current,
                                            mtx_00, 
                                            dist_00, 
                                            flags) 
            
            # Re-project the points
            rep_imgpt = cv.projectPoints(objp_current, 
                                         rvecs, 
                                         tvecs, 
                                         mtx_00, 
                                         dist_00)
            
            # Reshape arrays for ease
            rep_imgpt = np.ascontiguousarray(rep_imgpt[0]).reshape((4,2))
            corners_current = np.ascontiguousarray(corners_current).reshape((4,2))
            
            # Reprojection error between detected corners and reprojected corners
            res = (rep_imgpt - corners_current)**2
            rep_er = (np.sqrt(res[:,0] + res[:,1]))
            calc_MSE = (1/(2*4*no_images))*np.sum(rep_er**2)
            MSE_v.append(calc_MSE)
            
            if i == unused:
                corner1+=14
                unused+=5
            else:
                corner1 +=2
        else:
            continue
# Calculate MAD across all MSE_virtual
MSE_mad = sts.median_abs_deviation(np.array(MSE_v))
# Calculate detector variance 
sigma_d_sq = 4*MSE_mad
# Calculate systematic error contribution
eps_bias_sq = np.max([((ret_00**2)/(1 - (no_extrinsics/no_corners_total)))-sigma_d_sq, 0])
# Calculate bias ratio
bias_ratio = (eps_bias_sq*(1 - (no_extrinsics/no_corners_total)))/(ret_00**2)

RMS_mad = np.sqrt(MSE_mad)



# Needed to calculate Z and tilt angle A to preserve WD for camera calibration

with open("data/sys_config.json", 'r') as fp:
    sys_data = json.load(fp)   

XYZ_OFFSET =  np.array(sys_data["xyz_offset"]).squeeze()
WORKING_DISTANCE = sys_data["working_distance"]
ANGLE_OFFSET = sys_data["camera_offsets"]
LIMITS = sys_data["axis_range"] 


A = 45

point = [ 0, 0, 0 ]
rotation_angle = 67
rotated_point = [(point[0]*np.cos(rotation_angle) - 
                  point[1]*np.sin(rotation_angle)),
                 (point[0]*np.sin(rotation_angle) + 
                  point[1]*np.cos(rotation_angle)),
                  point[2]]

z_machine = ((WORKING_DISTANCE * 
              np.sin((A + ANGLE_OFFSET["system"])/(180/np.pi))) - 
             XYZ_OFFSET[2] + rotated_point[2])

z_new = WORKING_DISTANCE*np.cos(90-A)
