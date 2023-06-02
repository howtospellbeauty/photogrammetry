# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 11:17:55 2022

@author: Emma Woods

This is the motion characterisation module where the motion correction 
variables are calculated. 

The current implementation takes checkerboard images along / around each axis
so that a correction vector can be calculated.

Note: The motion analysis class needs further work, there are a few bugs still
that I've been unable to fix. It might be worth going back to Danny's original
code in Software Development - Documents\Algorithm development\PM\P2_Control


"""

# External Libraries
import json
import copy
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Taraz Modules
import system_control as sc
import camera_characterisation as cc
import mathematical_functions as mf





# %% motion_capture function [Creates motion_capture.json]

def motion_capture(board, camera, Waylands = False):
    """
    Move to nominal location where checkerboard should be in the centre 
    of the frame. Save coordinates if checkerboard location is returned

    Move to fixed locations along X, Y, Z, A, B axes and save
    coordinates if checkerboard location is returned. 

    Move to random locations and save coordinates if checkerboard 
    location is returned. 

    Use cv.CalibrateCamera to return intrinsic matrix and R and T 
    vectors for random locations and save. 

    Use cv.solvePnPRansac to return R and T vectors for nominal and axis 
    locations and save. 

    Output: motion_capture.json to be used in MotionAnalysis.

    """

    # ---------------------- Fixed settings ---------------------------

    # Number of random locations to sample
    random_locations = 1 #30

    # Spatial resolution for each axis
    resolution = np.array([15, 15, 15, 15, 20])
    
    # Load axis configuration from system config
    if Waylands == True:            
        with open("data/sys_config_waylands.json", 'r') as fp:
            sys_data = json.load(fp)
            
        # select which camera to take images from    
        cam = 1 
    else:
        with open("data/sys_config.json", 'r') as fp:
            sys_data = json.load(fp) 
            
        # select which camera to take images from     
        cam = 0
    
    # range for each axis
    axis_range = sys_data["axis_range"]
    
    # nominal location that should be able to see the centre
    nominal_loc = np.array(sys_data["nominal_loc"]).squeeze()



    # -------------------- Create Data Storage ------------------------

    # Create dictionary for axis image points
    axis_impts = {
        "x-axis": [],
        "y-axis": [],
        "z-axis": [],
        "a-axis": [],
        "b-axis": []
    }

    # Create dictionary objects for calibration
    blank_axis = {"req_loc": [],
                  "R": [],
                  "T": []}

    axis_data = {"x-axis": copy.deepcopy(blank_axis),
                 "y-axis": copy.deepcopy(blank_axis),
                 "z-axis": copy.deepcopy(blank_axis),
                 "a-axis": copy.deepcopy(blank_axis),
                 "b-axis": copy.deepcopy(blank_axis)
                 }

    motion_data = {
        "axis_data": axis_data,
        "nominal": copy.deepcopy(blank_axis),
        "random": copy.deepcopy(blank_axis)
    }

    # ------------------ Calibration Information -----------------------
    
    # Create calibration artefact instance
    artefact = cc.CalibrationArtefacts()

    # Pre-allocate image points and world points
    impts = []
    wrldpts = []


    # ------------ Capture image at nominal location ------------------

    sc.DuetBoard.move(nominal_loc, board)
    image = sc.Camera.camera_capture(camera)[cam]

    ret, corners = artefact.detect_checkerboard(image)

    if ret:
        motion_data["nominal"]["req_loc"].append(nominal_loc.tolist())
        nominal_impts = corners
        # plt.imshow(image)
        # plt.plot(corners[:, 0, 0], corners[:, 0, 1], 'rx')
        # plt.show()
    
    # -------------- Capture images for each axis ---------------------

    axis_list = list(motion_data["axis_data"].keys())

    for k in range(len(axis_list)):

        # Current axis locations
        axis_locations = np.linspace(axis_range["min"][k],
                                     axis_range["max"][k],
                                     resolution[k])
        # Generate location
        location = copy.copy(nominal_loc)

        # Run through each sample location
        for n in range(resolution[k]):

            location[k] = axis_locations[n]

            sc.DuetBoard.move(location, board)
            
            
            image = sc.Camera.camera_capture(camera)[cam]

            ret, corners = artefact.detect_checkerboard(image)

            if ret:
                (motion_data
                 ["axis_data"]
                 [axis_list[k]]
                 ["req_loc"].append(location.tolist()))

                axis_impts[axis_list[k]].append(corners)
                
    
    print("Axis locations Done")
    # ------ Capture random locations to improve calibration ----------
    #  Make this dynamic?

    # Track repro error
    error_track = []

    # Update range to max/min values where checkerboard was resolvable
    ## This may not be the best method: see commented out random location 
    ## generation on lines 233 to 236
    range_min = []
    range_max = []
    for k in range(5):
        range_min.append(
            motion_data["axis_data"][axis_list[k]]["req_loc"][0][k])
        range_max.append(
            motion_data["axis_data"][axis_list[k]]["req_loc"][-1][k])

    Range = {"min": range_min,
             "max": range_max }

    # Calibration count
    calibration_count = 0
    while calibration_count < random_locations:

        # Generate location

        # # rng = np.random.default_rng(2022)
        rng = np.random.default_rng()
        # Create random additonal vector        
        # rand_add = (np.array(axis_range["min"]) + rng.random(5) *
        #            (np.array(axis_range["max"]) - np.array(axis_range["min"])))
        # location = nominal_loc+rand_add
        location = rng.uniform(Range["min"],Range["max"])

        
        
        sc.DuetBoard.move(location, board)
        image = sc.Camera.camera_capture(camera)[cam]

        ret, corners = artefact.detect_checkerboard(image)

        if ret:
            motion_data["random"]["req_loc"].append(location.tolist())
            calibration_count += 1
            impts.append(corners)
            wrldpts.append(artefact.objp)

            if calibration_count >= 5:  # WHY ONLY WHEN ABOVE 5?
                cal_flag = cv.CALIB_FIX_ASPECT_RATIO
                rms, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
                                                            wrldpts,
                                                            impts,
                                                            image.shape[::-1],
                                                            None,
                                                            None,
                                                            flags=cal_flag)

                # print(ret_camera)
                error_track.append(rms)

    error_track = np.array(error_track)
    plt.plot(error_track)
    # plt.show()

    print("random locations complete")

    # ------------------- Camera calibration --------------------------

    """ Use the random locations to work out the camera intrinsic 
    matrix """

    # Calibrate camera
    # TODO: This shouldn't recalibrate the cameras again, we should be using
    # the camera calibration already obtained rather than having different 
    # ones used thoughout the system 
    cal_flag = cv.CALIB_FIX_ASPECT_RATIO
    rms, mtx, dist, rvecs, tvecs = cv.calibrateCamera(wrldpts,
                                                      impts,
                                                      image.shape[::-1],
                                                      None,
                                                      None,
                                                      flags=cal_flag)

    motion_data["random"]["R"] = []
    motion_data["random"]["T"] = []
    for k in range(len(rvecs)):
        motion_data["random"]["R"].append(rvecs[k].tolist())
        motion_data["random"]["T"].append(tvecs[k].tolist())

    print("Camera re-projection is " + "{:.2f}".format(rms) + " pixels")

    # -------------------  Extract camera pose ------------------------

    """Use the acquired camera matrix to extract the camera pose for 
    each position tested along the axes and the nominal location. """

    # Get nominal information
    ret, rvecs, tvecs, inliers = cv.solvePnPRansac(artefact.objp,
                                                   nominal_impts,
                                                   mtx,
                                                   dist)

    motion_data["nominal"]["R"] = rvecs.tolist()
    motion_data["nominal"]["T"] = tvecs.tolist()

    for k in range(len(axis_list)):

        motion_data["axis_data"][axis_list[k]]["R"] = []
        motion_data["axis_data"][axis_list[k]]["T"] = []
        for m in range(len(axis_impts[axis_list[k]])):

            # Find the rotation and translation vectors.
            ret, rvecs, tvecs, inliers = cv.solvePnPRansac(
                                                artefact.objp,
                                                axis_impts[axis_list[k]][m],
                                                mtx,
                                                dist)

            motion_data["axis_data"][axis_list[k]
                                     ]["R"].append(rvecs.tolist())
            motion_data["axis_data"][axis_list[k]
                                     ]["T"].append(tvecs.tolist())

    # ------------------------ Export Data ----------------------------
    # Export data
    with open('data/motion_capture.json', 'w') as fp:
        json.dump(motion_data, fp, indent=4)

    return motion_data

# %% motion_analysis class [Creates motion_correction.json]


class MotionAnalysis:
    """
    Functions for the processing and evaluation of the motion system
    
    Note: I don't know how to use the resulting information
    """

    # ---------------- Methods called (in order) ----------------------

    def calculate_locations(self, motion_data):
        # Extract list of axes
        axes_list = list(motion_data["axis_data"].keys())
        # Camera locations
        camera_location = []
        # Extract location
        for k in range(len(axes_list)):
            motion_data["axis_data"][axes_list[k]]["loc"] = []
            axis_count = len(motion_data["axis_data"][axes_list[k]]["R"])
            for n in range(axis_count):
                R = copy.copy(mf.Extrinsics.vec2r(
                    np.array(motion_data
                             ["axis_data"]
                             [axes_list[k]]
                             ["R"]
                             [n])))
                T = copy.copy(np.array(motion_data
                                       ["axis_data"]
                                       [axes_list[k]]
                                       ["T"]
                                       [n]))

                # flip reference frame from checkerboard to cam
                loc = np.dot(np.linalg.inv(R), T).reshape((1, 3))
                motion_data["axis_data"][axes_list[k]
                                         ]["loc"].append(loc.tolist())

                camera_location.append(loc)

        # nominal data
        motion_data["nominal"]["loc"] = []
        axis_count = len(motion_data["nominal"]["R"])
        R = copy.copy(mf.Extrinsics.vec2r(
            np.array(motion_data["nominal"]["R"])))
        T = copy.copy(np.array(motion_data["nominal"]["T"]))
        loc = np.dot(np.linalg.inv(R), T).reshape((1, 3))
        motion_data["nominal"]["loc"].append(loc.tolist())
        camera_location.append(loc)

        # random data
        motion_data["random"]["loc"] = []
        axis_count = len(motion_data["random"]["R"])
        for n in range(axis_count):
            R = copy.copy(mf.Extrinsics.vec2r(
                np.array(motion_data["random"]["R"][n])))
            T = copy.copy(np.array(motion_data["random"]["T"][n]))
            loc = np.dot(np.linalg.inv(R), T).reshape((1, 3))
            motion_data["random"]["loc"].append(loc.tolist())
            camera_location.append(loc)
        return motion_data, camera_location

    def level_centre(self, motion_data):
        # Extract table rotation locations
        rotation_locations = np.array(copy.deepcopy(
            motion_data["axis_data"]["b-axis"]["req_loc"])).squeeze()
        # Fit data to plane
        normal, res = mf.Fitting.plane_normal(rotation_locations[:,:3])
        # Rotation matrix for leveling
        R_level = mf.Fitting.rotate_data_xyz(normal, np.array([0, 0, 1]))
        # Apply rotation to data
        motion_data = self.apply_rotation(motion_data, R_level)
        rotation_locations = np.dot(R_level, rotation_locations.transpose())
        # Centre data
        xc, yc, r, residu = mf.Fitting.leastsq_circle(rotation_locations[0, :],
                                                      rotation_locations[1, :])
        T_centre = np.array([-xc, -yc, 0])
        # Apply rotation to data
        motion_data = self.apply_translation(
            motion_data, T_centre)
        for k in range(2):
            rotation_locations[k, :] += T_centre[k]
        return motion_data

    def orientate_x(self, motion_data):
        # Extract list of axes
        axes_list = list(motion_data["axis_data"].keys())

        # Extract data for x_axis
        k = 0

        # Camera locations
        camera_location = []
        axis_count = len(motion_data["axis_data"][axes_list[k]]["R"])
        for n in range(axis_count):
            camera_location.append(copy.copy(np.array(
                                             motion_data
                                             ["axis_data"]
                                             [axes_list[k]]
                                             ["loc"]
                                             [n])))

        camera_location = np.array(camera_location).squeeze()

        # Average X axis location
        x_direction = np.mean(camera_location, axis=0)
        rotation_angle = np.arctan(x_direction[1]/x_direction[0])
        R_x_fix = mf.Extrinsics.vec2r(
            np.array([0, 0, 1]) * -rotation_angle)
        motion_data = self.apply_rotation(motion_data, R_x_fix)

        return motion_data

    def xyz_characterisation(self, motion_data, motion_variables):
        # Extract list of axes
        axes_list = list(motion_data["axis_data"].keys())
        
        # Extract data for x,y,z axes
        for k in range(3):
            # Camera locations
            camera_location = []
            # Machine location
            machine_location = []
            axis_count = len(
                motion_data["axis_data"][axes_list[k]]["R"])
            for n in range(axis_count):

                camera_location.append(copy.copy(np.array(
                                                 motion_data
                                                 ["axis_data"]
                                                 [axes_list[k]]
                                                 ["loc"]
                                                 [n])))
                machine_location.append(copy.copy(np.array(
                                                  motion_data
                                                  ["axis_data"]
                                                  [axes_list[k]]
                                                  ["req_loc"]
                                                  [n]
                                                  [k])))

            camera_location = np.array(camera_location).squeeze()
            machine_location = np.array(machine_location).squeeze()

            # Camera and machine steps
            d_camera = camera_location[1:, :] - camera_location[:-1, :]
            d_machine = machine_location[1:] - machine_location[:-1]

            # Normalise
            for m in range(3):
                d_camera[:, m] /= d_machine
            vector = np.mean(d_camera, axis=0)
            motion_variables[axes_list[k]] = {"vector": vector.tolist()}

        return motion_variables

    def table_angle_characterisation(self, motion_data, motion_variables):
        # Extract list of axes
        axes_list = list(motion_data["axis_data"].keys())
        # Camera locations
        camera_location = []
        # Machine location
        machine_location = []
        axis_count = len(motion_data["axis_data"][axes_list[4]]["R"])
        for n in range(axis_count):
            camera_location.append(copy.copy(np.array(
                                                    motion_data
                                                    ["axis_data"]
                                                    [axes_list[4]]
                                                    ["loc"]
                                                    [n])))
            machine_location.append(copy.copy(np.array(
                                                      motion_data
                                                      ["axis_data"]
                                                      [axes_list[4]]
                                                      ["req_loc"]
                                                      [n]
                                                      [4])))

        camera_location = np.array(camera_location).squeeze()
        machine_location = np.array(machine_location).squeeze() / (180/np.pi)
        rho, phi = mf.Extrinsics.cart2pol(
            camera_location[:, 0], camera_location[:, 1])

        phi = phi + (((phi-np.pi) < (-np.pi)) * (2*np.pi))
        grad, intercept, res = mf.Fitting.line_fit(
            machine_location, phi)
        plt.plot(machine_location, phi, 'x')
        plt.xlabel("machine location - B axis")
        plt.ylabel("phi")
        plt.show()
        motion_variables[axes_list[4]] = {
            "Angle_correction": np.array([grad, intercept]).tolist()}
        return motion_variables

    def rotation_vectors(self, motion_data, motion_variables):
        # Extract list of axes
        axes_list = list(motion_data["axis_data"].keys())
        for k in range(3, 5):
            # Camera locations
            camera_orientation = []
            # Machine location
            machine_location = []
            axis_count = len(
                motion_data["axis_data"][axes_list[k]]["R"])
            for n in range(axis_count):
                camera_orientation.append(copy.copy(np.array(
                                                    motion_data
                                                    ["axis_data"]
                                                    [axes_list[k]]
                                                    ["R"]
                                                    [n])))
                machine_location.append(copy.copy(np.array(
                                                  motion_data
                                                  ["axis_data"]
                                                  [axes_list[k]]
                                                  ["req_loc"]
                                                  [n]
                                                  [k])))

            camera_orientation = np.array(camera_orientation).squeeze()
            machine_location = np.array(
                machine_location).squeeze() / (180/np.pi)

            # Extract rotation from stage
            orientation_change = np.zeros(
                (camera_orientation.shape[0]-1,
                 camera_orientation.shape[1]))

            vector = np.zeros(orientation_change.shape)
            angle = np.zeros(orientation_change.shape[0])

            for n in range(camera_orientation.shape[0]-1):
                R_A = mf.Extrinsics.vec2r(camera_orientation[n, :])
                R_B = mf.Extrinsics.vec2r(camera_orientation[n+1, :])
                R_diff = np.dot(R_B, np.linalg.inv(R_A))
                orientation_change = mf.Extrinsics.r2vec(R_diff)
                angle[n] = np.sqrt(np.sum(orientation_change**2))
                vector[n, :] = orientation_change / angle[n]

            rot_vector = np.mean(vector, axis=0)
            motion_variables[
                axes_list[k]]["rotation_vector"] = rot_vector.tolist()

        return motion_variables

    def nominal_update(self, motion_data, motion_variables):

        motion_variables["nominal"] = motion_data["nominal"]
        motion_variables["nominal"]["radius"] = np.sqrt(
            np.sum(np.array(motion_variables["nominal"]["loc"])**2))

        return motion_variables

    def rotational_offset(motion_data, motion_variables):

        # Camera locations
        camera_orientation = []

        # nominal rotation_vector
        R_nom = mf.Extrinsics.vec2r(
            np.array(motion_variables["nominal"]["R"]))

        # Machine location
        machine_location = []
        axis_count = len(motion_data["random"]["R"])
        for n in range(axis_count):
            camera_orientation.append(copy.copy(np.array(motion_data
                                   ["random"]
                                   ["R"]
                                   [n])))
            machine_location.append(copy.copy(np.array(
                                   motion_data
                                   ["random"]
                                   ["req_loc"]
                                   [n]
                                   [3:])))
        camera_orientation = np.array(camera_orientation).squeeze()
        machine_location = np.array(
            machine_location).squeeze() / (180/np.pi)

        # Extract rotation from stage
        vec_off = np.zeros(camera_orientation.shape)
        for n in range(camera_orientation.shape[0]):
            R_cam = mf.Extrinsics.vec2r(camera_orientation[n, :])
            A_value = machine_location[n, 0] - (motion_variables
                                                ["nominal"]
                                                ["req_loc"]
                                                [0]
                                                [3] / (180/np.pi))
            A_vector = np.array(motion_variables
                                ["a-axis"]
                                ["rotation_vector"])

            R_A = mf.Extrinsics.vec2r(A_vector * A_value)
            B_value = ((machine_location[n, 1] *
                       motion_variables
                       ["b-axis"]
                       ["angle_correction"]
                       [0]) +
                       motion_variables
                       ["b-axis"]
                       ["angle_correction"]
                       [1])

            B_vector = np.array(
                motion_variables["b-axis"]["rotation_vector"])
            R_B = mf.Extrinsics.vec2r(B_vector * B_value)
            R_est = np.dot(R_B, np.dot(R_A, R_nom))
            R_off = np.dot(R_cam, np.linalg.inv(R_est))
            vec_off[n, :] = mf.Extrinsics.r2vec(R_off)
        vec_off = np.mean(vec_off, axis=0)
        motion_variables["offset"]["rotation"] = vec_off.tolist()
        return motion_variables

    def tilt_offsets(self, motion_data, motion_variables):
        # Extract list of axes
        axes_list = list(motion_data["axis_data"].keys())
        # Extract data for Tilt axis
        k = 3
        # Camera locations
        camera_location = []
        # Machine location
        machine_location = []
        axis_count = len(motion_data["axis_data"][axes_list[k]]["R"])
        for n in range(axis_count):
            camera_location.append(copy.copy(np.array(
                                             motion_data
                                             ["axis_data"]
                                             [axes_list[k]]
                                             ["loc"]
                                             [n])))
            machine_location.append(copy.copy(np.array(
                                              motion_data
                                              ["axis_data"]
                                              [axes_list[k]]
                                              ["req_loc"]
                                              [n]
                                              [k])))

        camera_location = np.array(camera_location).squeeze()
        machine_location = np.array(
            machine_location).squeeze() / (180/np.pi)
        # Fit data to a circle
        xc, zc, r, residu = mf.Fitting.leastsq_circle(
            camera_location[:, 0],
            camera_location[:, 2])

        motion_variables["a-axis"]["radius"] = r

        # Centre of rotation
        tilt_centre = np.array([xc, np.mean(camera_location[:, 1]), zc])

        # Angle with respect to centre
        rho, phi = mf.Extrinsics.cart2pol(
            camera_location[:, 0]-xc,
            camera_location[:, 2]-zc)

        # Relationship between machine and camera tilted location
        grad, intercept, res = mf.Fitting.line_fit(
            machine_location, phi)
        (motion_variables["a-axis"]
                         ["angle_correction"]) = np.array(
                                                 [grad, intercept]).tolist()

        # Extract XYZ offset
        xyz_offset = (tilt_centre -
                      (motion_data["axis_data"]
                       [axes_list[k]]["req_loc"][0][0] *
                       np.array(motion_variables["x-axis"]["vector"])) +
                      (motion_data["axis_data"]
                       [axes_list[k]]["req_loc"][0][1] *
                       np.array(motion_variables["y-axis"]["vector"])) +
                      (motion_data["axis_data"]
                       [axes_list[k]]["req_loc"][0][2] *
                       np.array(motion_variables["z-axis"]["vector"])))

        motion_variables["offset"]["translation"] = xyz_offset.tolist()
        return motion_variables

    def elevation_relation(self, motion_data, motion_variables):

        # Extract list of axes
        axes_list = list(motion_data["axis_data"].keys())

        # Extract data for Tilt axis
        k = 3

        # Camera locations
        angle = []

        # Machine location
        machine_location = []
        axis_count = len(motion_data["axis_data"][axes_list[k]]["R"])

        for n in range(axis_count):
            camera_orientation = (copy.copy(np.array(motion_data
                                                     ["axis_data"]
                                                     [axes_list[k]]
                                                     ["R"]
                                                     [n])))

            machine_location.append(copy.copy(np.array(motion_data
                                                       ["axis_data"]
                                                       [axes_list[k]]
                                                       ["req_loc"]
                                                       [n]
                                                       [k])))

            R_camera = mf.Extrinsics.vec2r(camera_orientation)
            Camera_view_vector = np.dot(
                np.linalg.inv(R_camera),
                np.array([0, 0, 1]))
            angle.append(np.arctan(
                         Camera_view_vector[2]/Camera_view_vector[0]))
        angle = np.array(angle).squeeze()
        machine_location = np.array(
            machine_location).squeeze() / (180/np.pi)

        # Relationship between machine and camera tilted location
        grad, intercept, res = mf.Fitting.line_fit(
            machine_location, angle)
        motion_variables["elevation"] = [grad, intercept]
        return motion_variables

    # ------------------ Used in Above Methods ------------------------
    # ----------------- (not called externally) -----------------------

    def apply_rotation(self, motion_data, R):
        # Extract list of axes
        axes_list = list(motion_data["axis_data"].keys())
        # Apply rotation to data
        for k in range(len(axes_list)):
            axis_count = len(
                motion_data["axis_data"][axes_list[k]]["R"])
            for n in range(axis_count):
                R_orig = mf.Extrinsics.vec2r(np.array(motion_data
                                                      ["axis_data"]
                                                      [axes_list[k]]
                                                      ["R"]
                                                      [n]))
                R_new = mf.Extrinsics.r2vec(np.dot(R, R_orig))
                motion_data["axis_data"][axes_list[k]
                                         ]["R"][n] = R_new.tolist()
                T_orig = np.array(motion_data
                                  ["axis_data"]
                                  [axes_list[k]]
                                  ["T"][n]).squeeze()
                T_new = np.dot(R, T_orig)
                motion_data["axis_data"][axes_list[k]
                                         ]["T"][n] = T_new.tolist()
                loc_orig = np.array(motion_data
                                    ["axis_data"]
                                    [axes_list[k]]
                                    ["loc"]
                                    [n])
                loc_new = np.dot(R, loc_orig.transpose()).transpose()
                motion_data["axis_data"][axes_list[k]
                                         ]["loc"][n] = loc_new.tolist()
                
        # nominal data
        axis_count = len(motion_data["nominal"]["R"])
        R_orig = mf.Extrinsics.vec2r(
            np.array(motion_data["nominal"]["R"]))
        R_new = mf.Extrinsics.r2vec(np.dot(R, R_orig))
        motion_data["nominal"]["R"] = R_new.tolist()
        T_orig = np.array(motion_data["nominal"]["T"]).squeeze()
        T_new = np.dot(R, T_orig)
        motion_data["nominal"]["T"] = T_new.tolist()
        loc_orig = np.array(motion_data["nominal"]["loc"])
        loc_new = np.dot(R, loc_orig.squeeze()).transpose()
        motion_data["nominal"]["loc"] = loc_new.tolist()
        
        # random data
        axis_count = len(motion_data["random"]["R"])
        for n in range(axis_count):
            R_orig = mf.Extrinsics.vec2r(
                np.array(motion_data["random"]["R"][n]))
            R_new = mf.Extrinsics.r2vec(np.dot(R, R_orig))
            motion_data["random"]["R"][n] = R_new.tolist()
            T_orig = np.array(motion_data["random"]["T"][n]).squeeze()
            T_new = np.dot(R, T_orig)
            motion_data["random"]["T"][n] = T_new.tolist()
            loc_orig = np.array(motion_data["random"]["loc"][n])
            loc_new = np.dot(R, loc_orig.transpose()).transpose()
            motion_data["random"]["location"][n] = loc_new.tolist()
        return motion_data

    def apply_translation(self, motion_data, T):
        # Extract list of axes
        axes_list = list(motion_data["axis_data"].keys())
        # Apply rotation to data
        for k in range(len(axes_list)):
            axis_count = len(motion_data
                             ["axis_data"]
                             [axes_list[k]]
                             ["R"])

            for n in range(axis_count):
                T_orig = np.array(motion_data
                                  ["axis_data"]
                                  [axes_list[k]]
                                  ["T"][n]).squeeze()
                T_new = T_orig + T
                motion_data["axis_data"][axes_list[k]
                                         ]["T"][n] = T_new.tolist()
                loc_orig = np.array(motion_data
                                    ["axis_data"]
                                    [axes_list[k]]
                                    ["loc"]
                                    [n])
                loc_new = loc_orig + T
                motion_data["axis_data"][axes_list[k]
                                         ]["loc"][n] = loc_new.tolist()
        # nominal data
        axis_count = len(motion_data["nominal"]["R"])
        T_orig = np.array(motion_data["nominal"]["T"]).squeeze()
        T_new = T_orig + T
        motion_data["nominal"]["T"] = T_new.tolist()
        loc_orig = np.array(motion_data["nominal"]["loc"])
        loc_new = loc_orig + T
        motion_data["nominal"]["loc"] = loc_new.tolist()
        # random data
        axis_count = len(motion_data["random"]["R"])
        for n in range(axis_count):
            T_orig = np.array(motion_data["random"]["T"][n]).squeeze()
            T_new = T_orig + T
            motion_data["random"]["T"][n] = T_new.tolist()
            loc_orig = np.array(motion_data["random"]["loc"][n])
            loc_new = loc_orig + T
            motion_data["random"]["loc"][n] = loc_new.tolist()

        return motion_data

    # ------------------- NOT USED BUT USEFUL? ------------------------

    def nominal_extract(self, motion_variables):
        R = mf.Extrinsics.vec2r(
            np.array(motion_variables["nominal"]["R"]))
        T = np.array(motion_variables["nominal"]["T"])
        return R, T

    def visualise(self, motion_data):
        camera_location = self.extract_locations(motion_data)
        # Visualise all camera locations
        camera_location = np.array(camera_location)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(
            camera_location[:, 0, 0],
            camera_location[:, 0, 1],
            camera_location[:, 0, 2])

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

    def extract_locations(self, motion_data):
        # Extract list of axes
        axes_list = list(motion_data["axis_data"].keys())
        # Camera locations
        camera_location = []
        # Extract location
        for k in range(len(axes_list)):
            axis_count = len(
                motion_data["axis_data"][axes_list[k]]["R"])
            for n in range(axis_count):
                loc = copy.copy(np.array(motion_data
                                         ["axis_data"]
                                         [axes_list[k]]
                                         ["loc"]
                                         [n]))
                camera_location.append(loc)

        return camera_location

    def visualise_axis(self, motion_data, axis):
        camera_location = self.extract_axis(
            motion_data, axis)
        # Visualise all camera locations
        camera_location = np.array(camera_location)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(
            camera_location[:, 0, 0],
            camera_location[:, 0, 1],
            camera_location[:, 0, 2])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

    def extract_axis(self, motion_data, axis):
        # Extract list of axes
        axes_list = list(motion_data["axis_data"].keys())
        # Camera locations
        camera_location = []
        # Extract location
        axis_count = len(motion_data["axis_data"][axes_list[axis]]["R"])
        for n in range(axis_count):
            loc = copy.copy(np.array(motion_data
                                     ["axis_data"]
                                     [axes_list[axis]]
                                     ["loc"]
                                     [n]))
            camera_location.append(loc)
        return camera_location

    def machine2r(self, Machine, motion_variables):
        A_value = Machine[3] - (motion_variables["nominal"]
                                ["location"][0][3] / (180/np.pi))
        A_vector = np.array(
            motion_variables["a-axis"]["Rotation_vector"])
        R_A = mf.Extrinsics.vec2r(A_vector * A_value)
        B_value = ((Machine[4] *
                   motion_variables["b-axis"]["angle_correction"][0]) +
                   motion_variables["b-axis"]["angle_correction"][1])
        B_vector = np.array(motion_variables
                            ["b-axis"]
                            ["rotation_vector"])
        R_B = mf.Extrinsics.vec2r(B_vector * B_value)
        vec_off = np.array(motion_variables["offset"]["rotation"])

        # nominal rotation_vector
        R_nom = mf.Extrinsics.vec2r(np.array(motion_variables
                                             ["nominal"]
                                             ["R"]))
        R_cam = np.dot(mf.Extrinsics.vec2r(vec_off),
                       np.dot(R_B, np.dot(R_A, R_nom)))
        return R_cam

    def machine2loc(self, Machine, motion_variables):
        Tilt_angle = (((Machine[3] / (180/np.pi)) *
                       motion_variables["a-axis"]
                       ["angle_correction"][0]) +
                      motion_variables["a-axis"]
                      ["angle_correction"][1])
        Tilt_offset = mf.Extrinsics.pol2cart(motion_variables
                                             ["a-axis"]
                                             ["radius"],
                                             Tilt_angle)

        Table_angle = ((Machine[4] / (180/np.pi)) * motion_variables
                       ["b-axis"]
                       ["angle_correction"]
                       [0])
        Table_vector = np.array([0, 0, 1])
        Table_rotation = mf.Extrinsics.vec2r(Table_vector * Table_angle)
        location_camera = (motion_variables["offset"]["translation"] +
                           (Machine[0] * np.array(motion_variables
                                                  ["x-axis"]
                                                  ["vector"])) +
                           (Machine[1] * np.array(motion_variables
                                                  ["y-axis"]
                                                  ["vector"])) +
                           (Machine[2] * np.array(motion_variables
                                                  ["z-axis"]
                                                  ["vector"])) +
                           np.array([Tilt_offset[0], 0, Tilt_offset[1]]))
        location_camera = np.dot(Table_rotation, location_camera)
        return location_camera


#%%

def run_motion_calibration(board,camera, Waylands):
    """
    Capture the motion calibration data
    
    Run through the motion analysis pipeline

    """
    
    # ----------------- Capture motion calibration data --------------------

    # motion_data = motion_capture(board, camera, Waylands)
        
    with open('data/motion_capture.json', 'r') as fp:
        motion_data = json.load(fp)    
    # ------- Analyse Motion Data using MotionAnalysis pipeline ------------
    
    # Pre-allocate motion correction dictionary
    motion_variables = {"x-axis": {},
                        "y-axis": {},
                        "z-axis": {},
                        "a-axis": {},
                        "b-axis": {},
                        "nominal": {},
                        "offset": {},
                        "elevation": {}}
    
    analyse = MotionAnalysis()
    
    # Calculate locations
    motion_data, camera_location = analyse.calculate_locations(motion_data)

    # Level and centre
    motion_data = analyse.level_centre(motion_data)

    # Rotate the X axis into the correct location
    motion_data = analyse.orientate_x(motion_data)

    # Extract X, Y and Z vectors
    motion_variables = analyse.xyz_characterisation(
        motion_data, motion_variables)

    # Evaluate table angle
    motion_variables = analyse.table_angle_characterisation(
        motion_data, motion_variables)

    # Table rotation
    motion_variables = analyse.rotation_vectors(
        motion_data, motion_variables)

    # Extract nominal
    motion_variables = analyse.nominal_update(motion_data, motion_variables)

    # Rotational offset
    motion_variables = analyse.rotatational_offset(motion_data, motion_variables)

    # Tilt mechanism offset and XYZ offset
    motion_variables = analyse.tilt_offsets(motion_data, motion_variables)

    # Relationship between camera tilt and the elevation angle of the part
    motion_variables = analyse.elevation_relation(motion_data, motion_variables)

    # Save data
    with open('data/motion_correction.json', 'w') as fp:
        json.dump(motion_variables, fp, indent=4)
 