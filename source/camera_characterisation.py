# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 16:13:11 2022

@author: EmmaWoods

Stereo-camera calibration process using images acquired using characterisation
module. 


"""

# External Modules
import os
import json
import cv2 as cv
import numpy as np
from PIL import Image
from pathlib import Path

# Taraz Modules
import system_control as sc

#%% Calibration Artefacts class

class CalibrationArtefacts:
    """ 
    Add more methods for different types of calibration artefacts 
    """
    
    def __init__(self, artefact_type = "checkerboard"):
        
        if artefact_type == "checkerboard":
            # Checkerboard properties
            # Grid size (count inner squares)
            self.grid = [11, 12]
            
            # Grid spacing (in mm)
            self.spacing = 6
            
            # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ...,(6,5,0)
            objp = np.zeros((self.grid[0]*self.grid[1],3), np.float32)
            objp[:,:2] = np.mgrid[0:self.grid[0],
                                  0:self.grid[1]].T.reshape(-1,2)
            self.objp = objp * self.spacing
            
            # resize ratio to reduce image size
            self.resize_ratio = 2**3
            
            # termination criteria
            self.criteria = (cv.TERM_CRITERIA_EPS +
                        cv.TERM_CRITERIA_MAX_ITER, 30, 0.001) 

    def detect_checkerboard(self, img, corner_detect = True):
        
        # resize image to speed up detection
        #(This speeds it up and is what Danny implemented, but I don't know
        # how it affects the final calibration result)
        new_image_size = (int((img.shape[1])/self.resize_ratio),
                          int((img.shape[0])/self.resize_ratio))
            
        smaller_image = cv.resize(img.astype("uint8"),
                                  new_image_size,
                                  interpolation=cv.INTER_AREA)
        
        # find chessboard corners for smaller image
        ret, corners = cv.findChessboardCorners(smaller_image,
                                               (self.grid[0],self.grid[1]),
                                               None)
        
        if corner_detect:
            # if found do refined corner detection
            if ret:
                # Refined corner detection
                corners_sub = cv.cornerSubPix(img.astype("uint8"),
                                              corners*self.resize_ratio,
                                              (11, 11),
                                              (-1, -1),
                                              self.criteria)
                return ret, corners_sub
            else:
                return ret, []
        else:
            return ret

#%%
class CameraCalibration:
    
    def __init__(self):
        
        # save directory
        self.save_dir = "data/calibration_images"
        
        # make save directory if doesn't exist
        if os.path.exists(self.save_dir) == False:
            os.makedirs(self.save_dir)    
        # clear old data if does exist
        else:    
            [f.unlink() for f in Path(self.save_dir).glob("*") if f.is_file()]
        
        # Number of random locations to sample
        self.random_locations = 20
        
        # initiate CalibrationArtefact class
        self.artefact = CalibrationArtefacts()
        

        
    def capture_camera_calibration_images(self, board, camera, 
                                          Waylands = False):
        """
        Capture checkerboard/calibration artefact images and store in a folder
        
        TODO: This is currently taking images within pre-defined limits for 
        each axis that were manually checked beforehand. This needs work. It 
        may end up with the user manually positioning the checkerboard using a 
        schematic provided by Taraz with the scan head at a specific position - 
        see technical information doc for more info. 
        
        TODO: Random locations in view of both camperas should also be captured 
        as a dataset for the stereo calibration.
    
        """
    
        # ---------------------- Fixed settings ---------------------------
    
        # Load information from system config file
        if Waylands == True:            
            with open("data/sys_config_waylands.json", 'r') as fp:
                sys_data = json.load(fp)   
        else:
            with open("data/sys_config.json", 'r') as fp:
                sys_data = json.load(fp) 
            
        XYZ_OFFSET =  np.array(sys_data["xyz_offset"]).squeeze()
        WORKING_DISTANCE = sys_data["working_distance"]
        ANGLE_OFFSET = sys_data["camera_offsets"]
        LIMITS = sys_data["calibration_axis_range"]
        CAMERA_NAMES = sys_data["camera_names"]

        coord_spacing = np.array([10, 10, 0, 1, 0])
                 
            
        # -------------------- Create Data Storage ------------------------
        
        # Arrays to store object and image points from all the images.
        self.objpoints = [] # 3d point in real world space
        self.imgpoints_00 = [] # 2d points in image plane
        self.imgpoints_01 = [] # 2d points in image plane
        
        
        # -------------- Capture images for each axis ---------------------
        
        print("Capturing calibration locations")

        X = np.arange(LIMITS["min"][0], LIMITS["max"][0], coord_spacing[0])
        Y = np.arange(LIMITS["min"][1], LIMITS["max"][1], coord_spacing[1])
        
        A = [35, 40, 45] # Tilt angle
        B = [0, 45]      # Rotation angle
        
        for k in range(coord_spacing[3]):
            
            tilt_angle = A[k]
            z_machine = ((WORKING_DISTANCE * 
                               np.sin((tilt_angle + 
                               ANGLE_OFFSET["system"])/(180/np.pi))) - 
                         XYZ_OFFSET[2])
            
            for x in range(len(X)):
                
                x_machine = X[x]
                
                for y in range(len(Y)):
                    
                    y_machine = Y[y]
                    
                    for b in range(len(B)):
                        
                        rot_angle = B[b]
                        
                        location = [x_machine, 
                                    y_machine, 
                                    z_machine, 
                                    tilt_angle, 
                                    rot_angle]
                        
                

                        sc.DuetBoard.move(location, board)
            
                        image = sc.Camera.camera_capture(camera)
                        ret00, corners00 = self.artefact.detect_checkerboard(
                                                                    image[0])
                        ret01, corners01 = self.artefact.detect_checkerboard(
                                                                    image[1])
                            
                        if ret00 & ret01:
                            
                            self.objpoints.append(self.artefact.objp)
                            self.imgpoints_00.append(corners00)
                            self.imgpoints_01.append(corners01)
                            
                            
                            im = Image.fromarray(image[0])
                            im.save(self.save_dir + 
                                    "/image_" + CAMERA_NAMES[0] +
                                      "_X" + str(location[0]) +
                                      "_Y" + str(location[1]) +
                                      "_Z" + str(location[2]) +
                                      "_A" + str(location[3]) +
                                      "_B" + str(location[4]) + ".jpg")
                            
                            im = Image.fromarray(image[1])
                            im.save(self.save_dir + 
                                    "/image_" + CAMERA_NAMES[1] +
                                      "_X" + str(location[0]) +
                                      "_Y" + str(location[1]) +
                                      "_Z" + str(location[2]) +
                                      "_A" + str(location[3]) +
                                      "_B" + str(location[4]) + ".jpg")
                            
                            
                    


        print("Calibration locations complete")        

    def stereo_camera_calibration(self,camera):
        """
        OpenCV pipeline for individual and stereo camera calibration

        """
        
        # load image to extract image size
        example_image = sc.Camera.camera_capture(camera)
        
        # calibrate camera 0 from cam 0 dataset
        ret_00, mtx_00, dist_00, rvecs_00, tvecs_00 = cv.calibrateCamera(
                                                self.objpoints, 
                                                self.imgpoints_00, 
                                                example_image[0].shape[::-1],
                                                None,
                                                None)
        # calibrate camera 1 from cam 1 dataset
        ret_01, mtx_01, dist_01, rvecs_01, tvecs_01 = cv.calibrateCamera(
                                                self.objpoints, 
                                                self.imgpoints_01, 
                                                example_image[0].shape[::-1],
                                                None,
                                                None)
        
        # Do stereo-calibration
        stereocalib_criteria = (cv.TERM_CRITERIA_MAX_ITER + 
                                cv.TERM_CRITERIA_EPS, 200, 1e-5)
        
        # Flags to fix intrinsic and aspect ratio (fx = fy)
        flags = cv.CALIB_FIX_ASPECT_RATIO  + cv.CALIB_FIX_INTRINSIC
        
        # image_points_00 and image_points_01 should be from a dataset where
        # the checkerboard is in view in each camera and different from the 
        # individual datasets used above. 
        ret, M1, d1, M2, d2, R, T, E, F = cv.stereoCalibrate(
                                                    self.objpoints, 
                                                    self.imgpoints_00,
                                                    self.imgpoints_01,
                                                    mtx_00, 
                                                    dist_00,
                                                    mtx_01,
                                                    dist_01, 
                                                    None,
                                                    stereocalib_criteria, 
                                                    flags)
        
        # store matrices in dictionaries
        camera_00 = { "intrinsic": M1.tolist(),
                      "distortion": d1.tolist() }
        camera_01 = { "intrinsic": M1.tolist(),
                      "distortion": d1.tolist() }
        
        stereo_calibration = { "camera_00": camera_00,
                                "camera_01": camera_01,
                                "rotation": R.tolist(),
                                "translation": T.tolist(),
                                "fundamental": F.tolist() }
        
        # save stereo calibration information in file 
        # TODO: remove _test from file name when no longer testing
        with open("data/stereo_calibration_test.json", "w") as outfile:
            json.dump(stereo_calibration, outfile, indent=4)
        
            
        # Calculate baseline from translation vector: 
        #(This is for scaling, potentially save in above stereo calib dict)
        baseline = np.sqrt(np.sum(T**2))
        print(baseline)
        
#%%

def run_camera_calibration(board,camera):
    
    # create camera calibration instance
    calib = CameraCalibration()
    # capture images
    calib.capture_camera_calibration_images(board, camera)
    # run calibration
    calib.stereo_camera_calibration(camera)
        