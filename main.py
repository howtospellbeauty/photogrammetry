# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 2023
@author: Wenwen

This is the main pipeline file to conduct Photogrammetry measurements. 
Four files are required to run the full pipeline: 
    system_control.py
        - controls duet board and cameras
        - live view options
    motion_characterisation.py 
        - generates coordinate positions 
        - analyses the stage
    camera_characterisation.py
        - calculates intrinsits and extrinsics of the system
    scan_object.py
        - creates photogrammetry scan path
        - operates the camera
        - takes images
    reconstruction.py
        - pre-process photogrammetry images
        - sets up openMVG data file
        - extracts features from images
        - reconstructs point clouds
    mathematical_functions.py
    
        
    
Other packages required are: 
    OpenCV - computer vision library for checkerboard characterisation
    Pypylon - library for basler camera controls
    OpenMVG - reconstructs the point clouds
    OpenMVS - densification of the point clouds  
 
"""

# External Libraries
import subprocess
from pypylon import genicam
from time import sleep
import os
import sys
sys.path.insert(1,'./source/')
sys.path.insert(1,'./data/')

# Taraz Modules
import system_control as sc
import camera_characterisation as cc
import motion_characterisation as mc
import scan_object as so
import photogrammetry as pg
import point_cloud_processing as pcp



#%%
## USER SETTINGS ## 

# Using the waylands system?
Waylands = True

# Calibrate system?
calibrate_stage = False
calibrate_camera = False

# Capture new data?
scan_part = True
scan_name = "Example Scan"
measurement_directory = "measurements/"

# Pre-process images?
undistort = False
mask = False

# Run reconstruction?
reconstruct = True

# Pre-process point cloud?
process = False

#%% 
## RUN SYSTEM ##


if calibrate_stage or calibrate_camera or scan_part:

    # Try to connect to cameras
    try:
        # ensure pylonviewer always closed:
        subprocess.call(["taskkill","/F","/IM","pylonviewer.exe"])
        
        # Camera exposure level
        expo = [61659, 36907]  #TODO: AUTOMATE
        # COM port for connection
        port = sc.list_ports()
        # Initialise Connections to Board and Camera(s)
        board, camera = sc.initialise_connection(2,expo,port)
    except UnboundLocalError:
        print("Hardware Not Connected")
        raise SystemExit
    except genicam.RuntimeException:
        print("Cameras Already Connected - Restarting Kernel...") 
        sleep(5)
        os._exit(00)

    # Try to run the process
    try:
        if calibrate_stage:
            print("Calibrating Motion Stages")
            mc.run_motion_calibration(board,camera, Waylands)
        if calibrate_camera:
            print("Calibrating Cameras")            
            cc.run_camera_calibration(board,camera)
                
        if scan_part:
            print("Scan Part")
            # initialise PathGeneration class
            # use of waylands only needed while two system variations in place
            path = so.PathGeneration(Waylands)
            
            # create machine coordinates
            machine, ref_cam = path.create_path()
            
            sleep(0.5) # pause to ensure first image captured correctly
            
            # run data collection
            so.scan_process(camera, 
                            board, 
                            machine, 
                            measurement_directory,
                            scan_name, 
                            Waylands)
            
    except KeyboardInterrupt:
        sc.close_connections(board,camera)
    finally:
        sleep(1)
        sc.close_connections(board,camera)
        print("Disconnected from Hardware")


if reconstruct:
    pg.run_reconstruction(measurement_directory, scan_name, 
                          undistort, mask, Waylands)
    
if process: 
   scaling_factor = pcp.process_point_cloud(measurement_directory, scan_name)
   
   