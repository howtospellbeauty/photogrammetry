# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 11:17:55 2022

@author: Emma Woods

This is the photogrammetry module for the PG reconstruction.

It can pre-process the photogrammetry images (undistort and mask) as well as
create the photogrammetry point cloud. All classes and functions have been well
commented, but refer to the planning and technical documents for more info.


"""

# external modules
import os
import json
import time
import fnmatch
import cv2 as cv
import subprocess
import numpy as np
from PIL import Image
import multiprocessing


#%%
class ImageProcessing():
    """
    Class to store image processing functions
    
    Since the init covers most of the same variables as the photogrammetry 
    class it may be better to combine the two.
    
    """
    def __init__(self, measurement_directory, filename, Waylands = False):
        
        # Load information from system config file
        if Waylands == True:            
            with open("data/sys_config_waylands.json", 'r') as fp:
                sys_data = json.load(fp)   
        else:
            with open("data/sys_config.json", 'r') as fp:
                sys_data = json.load(fp)

        self.camera_names = sys_data["camera_names"]
        
        # load camera data and extract parameters
        with open('data/stereo_calibration.json', 'r') as fp:
            camera_calib = json.load(fp)

        self.mtx_00 = np.array(
            camera_calib[self.camera_names[0]]["intrinsic"])
        self.dist_00 = np.array(
            camera_calib[self.camera_names[0]]["distortion"][0])  

        self.mtx_01 = np.array(
            camera_calib[self.camera_names[1]]["intrinsic"])
        self.dist_01 = np.array(
            camera_calib[self.camera_names[1]]["distortion"][0])  
        
        # set up file directories
        self.base_dir = './'+measurement_directory+filename
        if os.path.exists(self.base_dir) == False:
            raise ValueError("No scan data")
        
        self.input_dir = self.base_dir + "/images/raw/"
        os.makedirs(self.input_dir, exist_ok=True)
        
        self.undistort_dir = self.base_dir + "/images/undistorted/"
        os.makedirs(self.undistort_dir, exist_ok=True)
        
        self.masked_dir = self.base_dir + "/images/masked/"
        os.makedirs(self.masked_dir, exist_ok=True)
        
        self.no_imgs = len(fnmatch.filter(os.listdir(self.input_dir), '*.jpg'))  
        
        
    def undistort_images(self):
        """
        Undistort photogrammetry images using camera distortion parameters.
        Uses openCV pipeline - see online documentation
        
        """
        
        tic = time.perf_counter() # only needed for development to track times
        
        img_names = os.listdir(self.input_dir)
        for n in range(self.no_imgs):
            
            current_image = cv.cvtColor(cv.imread(self.input_dir + 
                                                  img_names[n]),
                                        cv.COLOR_BGR2GRAY)
            img_sze = current_image.shape[::-1]
            
            if img_names[n][6:15] == self.camera_names[0]:
                mtx = self.mtx_00
                dist = self.dist_00

            elif img_names[n][6:15] == self.camera_names[1]:
                mtx = self.mtx_01
                dist = self.dist_01

            new_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, 
                                                        dist, 
                                                        img_sze, 
                                                        1, 
                                                        img_sze)

            new_img = cv.undistort(current_image, mtx, dist, None, new_mtx)
            img = Image.fromarray(new_img)
            img.save(self.undistort_dir + 
                     img_names[n].split('.')[0] + 
                     "_undistorted.jpg")
            
        # only needed for development to track times
        toc = time.perf_counter()
        print(f"Undistort in {(toc - tic)/60:0.0f} m {(toc - tic)%60:0.0f} s")
        
    def mask_images(self, undistort):
        """
        Mask photogrammetry images to remove background.
        Uses openCV pipeline - see online documentation
        """
        
        tic = time.perf_counter() # only needed for development to track times
        
        # select input directory
        if undistort:
            directory = self.undistort_dir
        else:
            directory = self.input_dir 
            
        img_names = os.listdir(directory)
        for n in range(self.no_imgs):

            img = cv.imread(directory + img_names[n])
            gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
            
            #TODO: dev work on thresholding numbers-adaptive or static?
            # is this adaptable to different lighting conditions and parts?
            
            bilat = cv.bilateralFilter(gray,5,75,75) 
            _,thresh = cv.threshold(bilat, 
                                    np.mean(bilat), 
                                    255, 
                                    cv.THRESH_BINARY_INV)
            edges = cv.dilate(cv.Canny(thresh,0,255),None)
            
            cnt = sorted(cv.findContours(edges, 
                                         cv.RETR_LIST, 
                                         cv.CHAIN_APPROX_SIMPLE)[-2], 
                         key=cv.contourArea)[-1]
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            mask = cv.drawContours(mask, [cnt],-1, (255), -1)
            
            # Dilate mask
            kernel = np.ones((5,5), np.uint8)
            mask = cv.dilate(mask, kernel, iterations=15)

            # Add mask to image
            dst = cv.bitwise_and(img, img, mask=mask)
            segmented = cv.cvtColor(dst, cv.COLOR_BGR2RGB)
            
            img = Image.fromarray(segmented)
            img.save(self.masked_dir + 
                     img_names[n].split('_undistorted')[0] + 
                     "_processed.jpg")
        
        # only needed for development to track times
        toc = time.perf_counter()
        print(f"Mask in {(toc - tic)/60:0.0f} m {(toc - tic)%60:0.0f} s")
        

class PGReconstruction():
    
    def __init__(self, measurement_directory, filename, 
                 undistort, mask, Waylands ,**kwargs):
        
        # define openMVG paths
        self.OPENMVG_SFM_BIN = "source/openMVG/"
        self.OPENMVS_BIN = "source/openMVS/"
        
        # Load information from system config file
        if Waylands == True:            
            with open("data/sys_config_waylands.json", 'r') as fp:
                sys_data = json.load(fp)   
        else:
            with open("data/sys_config.json", 'r') as fp:
                sys_data = json.load(fp)
            
        self.camera_names = sys_data["camera_names"]
        self.camera_model = "1" # based on whether images have been undistorted
        
        # detect number of threads available for feature detection
        self.threads = str(multiprocessing.cpu_count())
        
        # this currently isn't used, needs further understanding of use of 
        # original file
        if "Original_file" in kwargs:
            self.original_file = kwargs["original_file"]
            
        # define all other directories
        self.base_dir = './'+measurement_directory+filename
        os.makedirs(self.base_dir, exist_ok=True)
        
        # select input directory based on pre-processing requirements
        if mask:
            self.input_dir = self.base_dir + "/images/masked"
            os.makedirs(self.input_dir, exist_ok=True)
        elif undistort:
            self.input_dir = self.base_dir + "/images/undistorted"
            os.makedirs(self.input_dir, exist_ok=True)
        else:
            self.input_dir = self.base_dir + "/images/raw"
            os.makedirs(self.input_dir, exist_ok=True)
        
        self.output_dir = self.base_dir + "/output"
        os.makedirs(self.output_dir, exist_ok=True)

        self.matches_dir = self.output_dir + "/matches"
        os.makedirs(self.matches_dir, exist_ok=True)

        self.reconstruction_dir = self.output_dir + "/reconstruction"
        os.makedirs(self.reconstruction_dir, exist_ok=True)
        
        self.dense_dir = self.output_dir + "/densify"
        os.makedirs(self.dense_dir, exist_ok=True)

        self.CAMERA_SENSOR_WIDTH_DIRECTORY = "data/"
        self.camera_file_params = os.path.join(
                                     self.CAMERA_SENSOR_WIDTH_DIRECTORY, 
                                     "sensor_width_database.txt")
        
        
        # nearest neighbour distance for matching
        self.nearest_neighbour_distance = "0.8" # Default value is 0.8
        
        # residual threshold for robust estimation
        self.residual_threshold = "2.0" # Default value is 4.0
        
        
                          
    def initial_listing(self, calibration_file):
        """
        Creates the sfm_data.json file needed for openMVG reconstruction.
        
        Image listings contain image names, the camera model to use and camera
        sensor data.
        
        File is then edited to attribute correct camera intrinsics to each 
        image. (TODO: This is where the extrinsics data will also be added)
        
        Several camera models can be input, selection of which controlled
        in init method (self.camera_model). If images have been undistorted 
        camera model should be basic pinhole ("1").
        
        """
        print ("1. Create Image Listings File")
        pIntrisics = subprocess.Popen( [os.path.join(self.OPENMVG_SFM_BIN, 
                                        "openMVG_main_SfMInit_ImageListing"),  
                                        "-i", self.input_dir,
                                        "-o", self.matches_dir, 
                                        "-d", self.camera_file_params, 
                                        "-c", self.camera_model,
                                        "-g", "0"])
                                        # "-k", self.int_matrix_0])
        pIntrisics.communicate()
        pIntrisics.wait()
        
        # load data to edit file
        listings_file = self.matches_dir+"/sfm_data.json"   
        with open(listings_file, 'r') as fp:
            listings_data = json.load(fp)
        # Load calibration data
        with open(calibration_file, 'r') as fp:
            camera_calib = json.load(fp)
            
        # change intrinsic id  for each image based on which camera took image
        # remember to change index if image name structure changes           
        for n in range(len(listings_data["views"])):
            name = (listings_data["views"][n]
                    ["value"]["ptr_wrapper"]["data"]["filename"])
            if name[6:15] == self.camera_names[0]:
                (listings_data["views"][n]
                 ["value"]["ptr_wrapper"]["data"]["id_intrinsic"]) = 0
            elif name[6:15] == self.camera_names[1]:
                (listings_data["views"][n]
                  ["value"]["ptr_wrapper"]["data"]["id_intrinsic"]) = 1
                
        
        int_list = []
        # add intrinsics data to json
        for m in range(2):
            # values for intrinsics dictionary
            int_id = (listings_data["views"][-1]
                      ["value"]["ptr_wrapper"]["id"]) + 1 + m
            
            
            int_fx = camera_calib[self.camera_names[m]]["intrinsic"][0][0]
            int_cx = camera_calib[self.camera_names[m]]["intrinsic"][0][2]
            int_cy = camera_calib[self.camera_names[m]]["intrinsic"][1][2]

            # intrinsics dictionary for each camera model
            if self.camera_model == "1":
                # Basic pinhole model
                int_dict = {
                            "key": m,
                            "value": {
                                "polymorphic_id": 2147483649,
                                "polymorphic_name": "pinhole",
                                "ptr_wrapper": {
                                    "id": int_id,
                                    "data": {
                                        "width": 5472,
                                        "height": 3648,
                                        "focal_length": int_fx,
                                        "principal_point": [
                                            int_cx,
                                            int_cy]
                                        }
                                    }
                                }
                            }
            elif self.camera_model == "2":
                
                # pinhole with 3 radial distortion parameters
                
                int_r1 = camera_calib[self.camera_names[m]]["distortion"][0][0]
                int_r2 = camera_calib[self.camera_names[m]]["distortion"][0][1]
                int_r3 = camera_calib[self.camera_names[m]]["distortion"][0][2]
                
                int_dict = {
                            "key": m,
                            "value": {
                                "polymorphic_id": 2147483649,
                                "polymorphic_name": "pinhole_radial_k3",
                                "ptr_wrapper": {
                                    "id": 2147483769,
                                    "data": {
                                        "value0": {
                                            "width": 5472,
                                            "height": 3648,
                                            "focal_length": int_fx,
                                            "principal_point": [
                                                int_cx,
                                                int_cy
                                            ]
                                        },
                                        "disto_k3": [
                                            int_r1,
                                            int_r2,
                                            int_r3
                                                ]
                                            }
                                        }
                                    }
                                }
            elif self.camera_model == "3":
                
                # pinhole with 3 radial and 2 tangential distortion parameters
                
                int_r1 = camera_calib[self.camera_names[m]]["distortion"][0][0]
                int_r2 = camera_calib[self.camera_names[m]]["distortion"][0][1]
                int_r3 = camera_calib[self.camera_names[m]]["distortion"][0][2]
                int_t1 = camera_calib[self.camera_names[m]]["distortion"][0][3]
                int_t2 = camera_calib[self.camera_names[m]]["distortion"][0][4]
                
                int_dict = {
                            "key": m,
                            "value": {
                                "polymorphic_id": 2147483649,
                                "polymorphic_name": "pinhole_brown_t2",
                                "ptr_wrapper": {
                                    "id": 2147483769,
                                    "data": {
                                        "value0": {
                                            "width": 5472,
                                            "height": 3648,
                                            "focal_length": int_fx,
                                            "principal_point": [
                                                int_cx,
                                                int_cy
                                            ]
                                        },
                                        "disto_t2": [
                                            int_r1,
                                            int_r2,
                                            int_r3,
                                            int_t1,
                                            int_t2
                                                ]
                                            }
                                        }
                                    }
                                }
            int_list.append(int_dict)
        listings_data["intrinsics"] = int_list
  
        with open(listings_file, 'w') as fp:
            json.dump(listings_data, fp, indent=4)


        
        
    def detect_features(self):
        """
        This uses the compute features function in the openMVG library.
        
        -p can be normal, high, or ultra depending on how many features
        you want to find, higher settings means more processing time
        
        """
        print ("2. Detect features")
        tic = time.perf_counter() # times are used for development only
        
        pFeatures = subprocess.Popen( [os.path.join(self.OPENMVG_SFM_BIN, 
                                             "openMVG_main_ComputeFeatures"),  
                                       "-i", self.matches_dir+"/sfm_data.json", 
                                       "-o", self.matches_dir, 
                                       #TODO: check if SIFT needs to be changed
                                       # for lisencing purposes
                                       "-m", "SIFT", 
                                       "-f" , "1", 
                                       "-p", "HIGH", 
                                       "-n", self.threads
                                       ])
        pFeatures.wait()
        toc = time.perf_counter()
        print(f"Features in {(toc - tic)/60:0.0f} m {(toc - tic)%60:0.0f} s")
        
    def sequential_reconstruction(self):
        
        """
        Sequential reconstruction as per the openMVG documentation. 
        First the matches are computed, followed by the reconstruction itself
        
        """
        
        print ("3. Compute matches")
        tic = time.perf_counter()  # times are used for development only
        
        
        pMatches = subprocess.Popen( [os.path.join(self.OPENMVG_SFM_BIN, 
                                            "openMVG_main_ComputeMatches"),  
                                      "-i", self.matches_dir+"/sfm_data.json", 
                                      "-o", self.matches_dir,
                                      "-f", "0",
                                      "-r", self.nearest_neighbour_distance,
                                      "-n", "ANNL2"
                                      #"-v", "10"
                                      ])
        pMatches.wait()
        toc = time.perf_counter()
        print(f"Matches in {(toc - tic)/60:0.0f} m {(toc - tic)%60:0.0f} s")
        tic = time.perf_counter()
        
        print ("4. Do Incremental/Sequential reconstruction") 
        #set manually the initial pair to avoid the prompt question
        pRecons = subprocess.Popen( [os.path.join(self.OPENMVG_SFM_BIN, 
                                           "openMVG_main_IncrementalSfM"),  
                                     "-i", self.matches_dir+"/sfm_data.json", 
                                     "-m", self.matches_dir, 
                                     "-o", self.reconstruction_dir] )
        pRecons.wait()
        toc = time.perf_counter()
        print(f"Recon in {(toc - tic)/60:0.0f} m {(toc - tic)%60:0.0f} s")
        
    def sequential2_reconstruction(self):
        """
        Currently unused, needs more research
        """
        print ("3. Compute matches")
        tic = time.perf_counter()
        pMatches = subprocess.Popen( [os.path.join(self.OPENMVG_SFM_BIN, 
                                            "openMVG_main_ComputeMatches"),  
                                      "-i", self.matches_dir+"/sfm_data.json", 
                                      "-o", self.matches_dir, 
                                      "-f", "0",
                                      "-n", "ANNL2",
                                      "-l", "Data/Matches_High.txt"] ) #CHECK WHERE FILE IS. 
        pMatches.wait()
        toc = time.perf_counter()
        print(f"Matches in {(toc - tic)/60:0.0f} m {(toc - tic)%60:0.0f} s")
        tic = time.perf_counter()
        print ("4. Do Incremental/Sequential reconstruction") 
        #set manually the initial pair to avoid the prompt question
        pRecons = subprocess.Popen( [os.path.join(self.OPENMVG_SFM_BIN, 
                                           "openMVG_main_IncrementalSfM2"),  
                                     "-i", self.Original_file, 
                                     "-m", self.matches_dir, 
                                     "-o", self.reconstruction_dir, 
                                     "-S", "EXISTING_POSE", 
                                     "-f", "NONE"] )
        pRecons.wait()
        toc = time.perf_counter()
        print(f"Recon in {(toc - tic)/60:0.0f} m {(toc - tic)%60:0.0f} s")
        
    def global_reconstruction(self):
        """
        Reconstruction for the global SfM pipeline
        - global SfM pipeline use matches filtered by the essential matrices
        - here we reuse photometric matches and perform only the essential 
          matrix filering
          
        Currently unused, needs more research
          
        """
        
        print ("3. Compute matches (for the global SfM Pipeline)")
        pMatches = subprocess.Popen( [os.path.join(self.OPENMVG_SFM_BIN, 
                                            "openMVG_main_ComputeMatches"),  
                                      "-i", self.matches_dir+"/sfm_data.json", 
                                      "-o", self.matches_dir, 
                                      "-r", self.nearest_neighbour_distance, 
                                      "-g", "e"] )
        pMatches.wait()
        print ("4. Do Global reconstruction")
        pRecons = subprocess.Popen( [os.path.join(self.OPENMVG_SFM_BIN, 
                                           "openMVG_main__GlobalSfM"),  
                                     "-i", self.matches_dir+"/sfm_data.json", 
                                     "-m", self.matches_dir, 
                                     "-o", self.reconstruction_dir, 
                                     "-S", "EXISTING_POSE", 
                                     "-f", "NONE"] )
        pRecons.wait()
        
    def reconstruction(self, method="sequential"):
        if method == "sequential":
            self.sequential_reconstruction()
        elif method == "sequential2":
            self.sequential2_reconstruction()
        elif method == "global":
            self.global_reconstruction()
        else:
            raise ValueError('Invalid reconstruction method.'+
                             ' Use: \n\tsequential (default) \n' + 
                             '\tsequential2 \n\tglobal')
        
    def colourise(self):
        """
        Used to find RGB colours for each point in the cloud.
        Useful for visualisation in cloud compare
        
        Poses are fully green (0 255 0).
        
        """
        print ("5. Colorize Structure")
        tic = time.perf_counter()
        pRecons = subprocess.Popen([os.path.join(self.OPENMVG_SFM_BIN, 
                                    "openMVG_main_ComputeSfM_DataColor"),  
                                    "-i", (self.reconstruction_dir+
                                           "/sfm_data.bin"), 
                                    "-o", os.path.join(self.reconstruction_dir,
                                                       "colorized.ply")] )
        pRecons.wait()
        toc = time.perf_counter()
        print(f"Colourized in {(toc - tic)/60:0.0f} m {(toc - tic)%60:0.0f} s")

    
    def robust(self):
        
        """
        Robust uses the known poses from prior reconstruction, this is
        potentially the function that will be used when we add the known poses
        in using the motion calibration information. 
        
        """
        
        print ("6. Structure from Known Poses (robust triangulation)")
        
        pRecons = subprocess.Popen( [os.path.join(self.OPENMVG_SFM_BIN, 
                                "openMVG_main_ComputeStructureFromKnownPoses"),  
                                "-i", self.reconstruction_dir+"/sfm_data.bin", 
                                "-m", self.matches_dir, 
                                "-o", os.path.join(self.reconstruction_dir,
                                                   "robust.ply"),
                                "-r", self.residual_threshold] )
        pRecons.wait()
        
    def densify(self):
        
        """
        Densitifation is used to add more points to the point cloud but it
        often doesn't work. However, it may not be needed as the normal point
        cloud may provide enough coverage to allow GD&T for most applications.
        
        """
        
        print ("7. Densification")
        pRecons = subprocess.Popen( [os.path.join(self.OPENMVG_SFM_BIN, 
                                          "openMVG_main_openMVG2openMVS"),  
                                     "-i", (self.reconstruction_dir+
                                            "/sfm_data.bin"), 
                                     "-d", self.dense_dir, 
                                     "-o", self.dense_dir + "/scene.mvs" ])
        pRecons.wait()
        pRecons = subprocess.Popen( [os.path.join(self.OPENMVS_BIN, 
                                                  "DensifyPointCloud"),  
                                     "-i", self.dense_dir+"/scene.mvs",
                                     #"-w", self.dense_dir,
                                     "-o", self.dense_dir + "/scene_dense.ply",
                                     "--dense-config-file", 
                                     "densify_options.txt"])
        pRecons.wait()
        print ("Densification complete")       

def run_reconstruction(measurement_directory, filename, 
                       undistort, mask, Waylands):
    
    # Pre-process images by undistorting and masking
    if undistort or mask:
        imgproc = ImageProcessing(measurement_directory, 
                                  filename, Waylands)
        if undistort:
            imgproc.undistort_images()
        if mask:
            imgproc.mask_images(undistort)
    
    # Camera calibration file
    calibration_file = 'data/stereo_calibration.json'
    
    # Original file - unsure what this is for
    original_file = ("measurements/Original_medium/output/"+
                      "reconstruction_sequential/sfm_data.bin")

    # Reconstruction method (sequential, sequential2, global)
    reconstruction_method = "sequential"
    
    # Reconstruction Pipeline  
    recon = PGReconstruction(measurement_directory, 
                            filename, undistort, mask, Waylands)
    
    recon.initial_listing(calibration_file)
    
    recon.detect_features()
    
    recon.reconstruction(reconstruction_method)
    
    recon.colourise()
    
    #TODO: Need to understand whether robust and densify are needed 
    # recon.robust()
    # recon.densify() 