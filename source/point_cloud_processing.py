# -*- coding: utf-8 -*-
"""
Created on Fri Apr 1 14:27:16 2022

@author: EmmaWoods

Module for post processing of point clouds
So far it consists of:
    - file type conversion from .ply to .txt 
    - rudamentary PG point cloud scaling
    
Future processing will likely include cleaning (noise and camera pose removal)
and point normal calculation.

"""

# External Libraries
import os
import numpy as np
import matplotlib.pyplot as plt

# Taraz Modules
import mathematical_functions as mf

#%%

class PointCloud():
    
    def __init__(self, measurement_directory, filename):

        self.OPENMVG_SFM_BIN = "source/openMVG/"
        self.OPENMVS_BIN = "source/openMVS/"
        
        # TODO: These should come from sys_config file
        self.cam0 = "camera_00"
        self.cam1 = "camera_01"

        self.base_dir = './'+measurement_directory+filename
        os.makedirs(self.base_dir, exist_ok=True)
        
        self.output_dir = self.base_dir + "/output"
        os.makedirs(self.output_dir, exist_ok=True)

        self.reconstruction_dir = self.output_dir + "/reconstruction"
        os.makedirs(self.reconstruction_dir, exist_ok=True)
        
        
    def point_cloud_file_type(self):
        # pc file name should change if not using the "colourized PC"
        pc_file = self.reconstruction_dir + "/colorized"
        filetype = ".ply"
        with open(pc_file+filetype, 'r') as pc:
            pc_data = pc.readlines()
            # remove header information
            pc_data = pc_data[10:]
        
        filetype = ".txt"
        with open(pc_file+filetype,'w') as pc:
            for item in pc_data:
                pc.write("%s\n" % item)


    def translate_data(xyz_data):
        # TODO: Move to mathematical_functions module
        
        x_c = np.mean(xyz_data[:,0])
        y_c = np.mean(xyz_data[:,1])
        T = np.array([x_c,y_c,0])
        
        translated_data = xyz_data - T
        return translated_data, T
    
    def reorient_scale_point_cloud(self):
        """
        This function reorients and scales the photogrammetry point cloud.
        
        It works by fitting a plane to one of the rings of camera poses and 
        aligning the plane to the z-axis. 
        
        Scaling is currently done using the baseline between the camera sensors
        but this should probably be changed to using the translation vector 
        from the stereo calibration. 
        See Algorithm development\PM\P2_Control\To_fix\Scale_factor.py
        
        Poses should probably be removed before outputting the final PC.
        """

        # Import the data 
        file = self.reconstruction_dir+'/colorized.txt' # change if not using colour  
        self.pc_data = np.loadtxt(file)
        
        cloud = self.pc_data[:,0:3]
        cloud_rgb = self.pc_data[:,3:]
        
        # TODO: num_poses should be carried through from scan_object module
        num_poses = 84
        cam_00_poses = self.pc_data[-num_poses*2:-num_poses,0:3]
        cam_01_poses = self.pc_data[-num_poses:,0:3]
        
        ring = 1
        # camera positions for a single 360 degree loop
        # This was used for testing to see how the scaling factor differed for
        # each ring, you'll likely want to take an average of all pose pair 
        # distances? There may be less of a difference poses prior to recon.
        if ring == 1:
            xyz_cam0 = cam_00_poses[0:28,0:3]
            xyz_cam1 = cam_01_poses[0:28,0:3] 
        elif ring == 2: 
            xyz_cam0 = cam_00_poses[28:56,0:3]
            xyz_cam1 = cam_01_poses[28:56,0:3]  
        elif ring == 3: 
            xyz_cam0 = cam_00_poses[56:,0:3]
            xyz_cam1 = cam_01_poses[56:,0:3] 
        
        # plot camera positions (for development)
        fig = plt.figure()
        ax = plt.axes(projection='3d') 
        ax.view_init(elev=0, azim=-90)
        
        ax.scatter(xyz_cam0[:,0],xyz_cam0[:,1],xyz_cam0[:,2],'blue')
        ax.scatter(xyz_cam1[:,0],xyz_cam1[:,1],xyz_cam1[:,2],'yellow')
        plt.xlabel('x')
        plt.ylabel('y') 
        
        
        # Rotate camera poses to align with z-axis normal
        # TODO: check if this always makes it "upside down" 
        cam0_data_new, Rxyz_cam0 = mf.Fitting.rotate_data_xyz(xyz_cam0)
        cam1_data_new, Rxyz_cam1 = mf.Fitting.rotate_data_xyz(xyz_cam1)
        
        # Translate camera poses to centre on (0,0)
        cam0_data_new, T00_cam0 = PointCloud.translate_data(cam0_data_new)
        cam1_data_new, T00_cam1 = PointCloud.translate_data(cam1_data_new)
        
        
        # Apply rotations & translations to all data
        cloud_rotated = np.array(np.dot(Rxyz_cam0,cloud.transpose()
                                              ).transpose()).squeeze()
        
        cloud_trans = cloud_rotated - T00_cam0
        
        
        # FIND SCALING FACTOR
        # distance between corresponding cam positions
        cam_xyz_differences = cam1_data_new - cam0_data_new
        cam_meas_distances = np.sqrt((cam_xyz_differences[:,0])**2 +
                                     (cam_xyz_differences[:,1])**2 +
                                     (cam_xyz_differences[:,2])**2)
        cam_meas_distance = np.mean(cam_meas_distances)
        
        # TODO: Change this to use translation vector between cameras
        d_centre_cam0 = 435.47*np.tan(25*np.pi / 180)    
        d_centre_cam1 = 435.47*np.tan(10*np.pi / 180) 
        d_cam0_cam1 = d_centre_cam0 + d_centre_cam1
        
        scaling_factor = d_cam0_cam1/cam_meas_distance
        
        # SCALE POINT CLOUD
        cloud_scaled = cloud_trans[:,0:3]*scaling_factor
        
              
        # reintroduce RGB values for saving
        self.processed_pc = np.concatenate((cloud_scaled,cloud_rgb),
                                            axis=1).tolist()
        
        
        # THIS IS JUST FOR VISUALISATION DURING DEVELOPMENT
        # Plot new data
        fig = plt.figure()
        ax = plt.axes(projection='3d') 
        ax.view_init(elev=0, azim=-90)
        ax.scatter(cam0_data_new[:,0],cam0_data_new[:,1],cam0_data_new[:,2],'blue')
        ax.scatter(cam1_data_new[:,0],cam1_data_new[:,1],cam1_data_new[:,2],'yellow')
        plt.xlabel('x')
        plt.ylabel('y')
        
        # returning the scaling factor is for dev work
        return scaling_factor


    def export_new_pc(self):
        # EXPORT THE DATA
        save_file = self.output_dir+'/processed_cloud.txt'  
        with open(save_file,'w') as pc:
            for item in self.processed_pc:
                pc.write(f'{str(item)[1:-1]}\n') 



#%%

def process_point_cloud(measurement_directory, scan_name):
    """
    Main function used to run the point cloud processing pipeline

    """
    
    # set up processing instance
    process = PointCloud(measurement_directory, scan_name) 

    try: 
        process.point_cloud_file_type()    
    except FileNotFoundError:
        print("Reconstruction Failed: no reconstructed point cloud.")
    
    # Returning scaling factor just for development
    scaling_factor = process.reorient_scale_point_cloud()
    
    process.export_new_pc()
    
    return scaling_factor
    
