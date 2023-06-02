# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 11:17:55 2022

@author: Emma Woods

The scan object module is used to generate the paths needed to scan an object,
and to capture and save the images from each location. 

This is likely where the extrinsics information for a particular scan will also
be saved. 

In the future, this module is likely to include the smart pathing algorithms
from MMT, but for now only a circular path is generated for a number of 
camera elevations. 



"""

#needed for path_generation
import numpy as np
import numpy.matlib as matlib
import json

import os
import system_control as sc
from PIL import Image

#%% Path generation from PG Toolbox

class PathGeneration():
    
    
    def __init__(self, Waylands = False):            
        
        # NOTE: Checking which P2 system is being used is purely a development
        # issue while there are two versions in use by the development team.
        # The only difference between the two files is the axis range and will
        # no longer be an issue if the Waylands system axes get upgraded.
        
        if Waylands == True:            
            with open("data/sys_config_waylands.json", 'r') as fp:
                sys_data = json.load(fp)   
        else:
            with open("data/sys_config.json", 'r') as fp:
                sys_data = json.load(fp) 
        
        self.XYZ_OFFSET =  np.array(sys_data["xyz_offset"]).squeeze()
        self.WORKING_DISTANCE = sys_data["working_distance"]
        self.ANGLE_OFFSET = sys_data["camera_offsets"] #I don't know where this 
                                                       # number is from
        self.LIMITS = sys_data["axis_range"]           
        
        
    def circular_path(self, tilt_angle_range, 
                      tilt_num, rotations_num, point):
        
        """
        Creates a number of rings of camera positions (elevations) based on the 
        number of tilts requested
        
        """
        machine = []
        ref_cam = []
        for tilt_angle in np.linspace(tilt_angle_range[0], 
                                      tilt_angle_range[1], 
                                      tilt_num):
            for rotation_angle in np.linspace(0, 360, rotations_num):
                
                valid_cam_0, machine_0 = self.cam2machine(tilt_angle, 
                                                          rotation_angle, 
                                                          point, 
                                                          0)
                valid_cam_1, machine_1 = self.cam2machine(tilt_angle, 
                                                          rotation_angle, 
                                                          point, 
                                                          1)

                if valid_cam_0:
                    machine.append(machine_0)
                    ref_cam.append(0)
                elif valid_cam_1:
                    machine.append(machine_1)
                    ref_cam.append(1)
                    
        machine = np.array(machine)
        ref_cam = np.array(ref_cam)
        return machine, ref_cam
    
    def create_path(self,pathtype = "circular"):
        """
        This function is to allow flexibility to introduce multiple path types
        such as the circular path, a saw tooth path, or a sinuoidal path. 
        
        It will also provide the ability to point towards a "smart" path once
        that is implemented

        """
        if pathtype == "circular":
            tilt_angle_range = [ 5, 45 ]
            tilt_num = 4
            rotations_num = 30
            point = [ 0, 0, 0 ] # point to rotate around
            machine, ref_cam = self.circular_path(tilt_angle_range,
                                                  tilt_num,
                                                  rotations_num,
                                                  point)
            return machine, ref_cam
    
    def cam2machine(self, tilt_angle, rotation_angle, point, camera):
        """
        function to convert between the camera angles and tilt to the 
        machine coordinates
        
        """
        
        rotated_point = [(point[0]*np.cos(rotation_angle) - 
                          point[1]*np.sin(rotation_angle)),
                         (point[0]*np.sin(rotation_angle) + 
                          point[1]*np.cos(rotation_angle)),
                          point[2]]
        
        A = tilt_angle - self.ANGLE_OFFSET[str(camera)]
        x_machine = ((self.WORKING_DISTANCE * 
                           np.cos((A + 
                           self.ANGLE_OFFSET["system"])/(180/np.pi))) - 
                     self.XYZ_OFFSET[0] + 
                     rotated_point[0])
        
        y_machine = rotated_point[1] - self.XYZ_OFFSET[1]
        z_machine = ((self.WORKING_DISTANCE * 
                           np.sin((A + 
                           self.ANGLE_OFFSET["system"])/(180/np.pi))) - 
                     self.XYZ_OFFSET[2] + 
                     rotated_point[2])
        machine = [ x_machine, y_machine, z_machine, A, rotation_angle]
        check = self.check_valid(machine)
        return check, machine
    
    def check_valid(self, machine):
        """
        Check whether machine location is witihn the axis limits

        """
        check_x = ((machine[0] >= self.LIMITS["min"][0]) * 
                   (machine[0] <= self.LIMITS["max"][0]))
        check_y = ((machine[1] >= self.LIMITS["min"][1]) * 
                   (machine[1] <= self.LIMITS["max"][1]))
        check_z = ((machine[2] >= self.LIMITS["min"][2]) * 
                   (machine[2] <= self.LIMITS["max"][2]))
        check_a = ((machine[3] >= self.LIMITS["min"][3]) * 
                   (machine[3] <= self.LIMITS["max"][3]))
        check_b = ((machine[4] >= self.LIMITS["min"][4]) * 
                   (machine[4] <= self.LIMITS["max"][4]))
        return check_x & check_y & check_z & check_a & check_b
    

    # ---------------------- NOT USED BUT USEFUL? ---------------------------
    
    def measurement_path(self, locations, tilt_range, point, methodology):
        machine = []
        ref_cam = []
        rotation_angle = np.linspace(0, 360, locations)
        levels = (tilt_range[1] - 
                  tilt_range[0])%self.step_angle
        
        if methodology == "sawtooth":
            tilts = np.linspace(tilt_range[0], 
                                     tilt_range[1], 
                                     levels)
            tilts = matlib.repmat(tilts, 
                                  1, 
                                  int(np.ceil(locations/levels))
                                  )[:locations].squeeze()

        elif methodology == "sine":
            period = 360 / (locations / (levels))
            tilts = tilt_range[0] + \
                np.sin((rotation_angle/period)*2*np.pi) * \
                    ((tilt_range[1] - tilt_range[0])/ 2)
        
        for k in range(locations):
            valid_cam_0, machine_0 = self.cam2machine(tilts[k], 
                                                      rotation_angle[k], 
                                                      point,
                                                      0)
            valid_cam_1, machine_1 = self.cam2machine(tilts[k], 
                                                      rotation_angle[k], 
                                                      point, 
                                                      1)
            if valid_cam_0:
                machine.append(machine_0)
                ref_cam.append(0)
            elif valid_cam_1:
                machine.append(machine_1)
                ref_cam.append(1)
        machine = np.array(machine)
        ref_cam = np.array(ref_cam)
        return machine, ref_cam
    

#%% 

def scan_process(camera, board, machine, 
                 measurement_directory, scan_name, Waylands = False):
    """ 
    Function controlling the pipline to scan an object.
    Take camera and duet board instances and machine coordinates created 
    from path generation. 
    
    Saves the images to the measurement directory (in the scan name folder).

    """
    
    # Load information from system config file
    if Waylands == True:            
        with open("data/sys_config_waylands.json", 'r') as fp:
            sys_data = json.load(fp)   
    else:
        with open("data/sys_config.json", 'r') as fp:
            sys_data = json.load(fp) 
        
    CAMERA_NAMES = sys_data["camera_names"]
    
    # create image save directory
    images_directory = measurement_directory + scan_name + "/images/raw"
    if os.path.exists(measurement_directory + scan_name) == False:
        os.makedirs(images_directory)

    # Capture images
    for k in range(machine.shape[0]):
        
        # Move to location
        sc.DuetBoard.move(machine[k,:], board)
            
        # Captue image
        Img = sc.Camera.camera_capture(camera)
        
        # Save all images
        for m in range(len(Img)):
            if m == 0:
                im = Image.fromarray(Img[m])
                im.save(images_directory + 
                        "/image_" + CAMERA_NAMES[m] +
                         "_" + str(k).zfill(4) + ".jpg")
            if m == 1:
                im = Image.fromarray(Img[m])
                im.save(images_directory + 
                        "/image_" + CAMERA_NAMES[m] +
                        "_" + str(k).zfill(4) + ".jpg")
                
                # If image names are ever changed, make sure to update the 
                # index searching in the PG reconstruction module.