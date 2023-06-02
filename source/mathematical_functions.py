# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 11:17:55 2022

@author: Emma Woods

This mathematical functions library is intended to be a common library of 
mathematical transforms and operations used throughout the sytem. 

Currently it isn't very flexible and there are mathematical functions within 
some modules regardless of this module existing. It could be worth collating
the mathematical operations used throughout the system and reconfiguring this
library and the code so that the inputs and outputs are common for all uses.

"""
# needed for extrinsics
import math
import numpy as np
from scipy.spatial.transform import Rotation

#needed for fitting
from scipy import optimize


#%% Rotation and translaion functions
class Extrinsics:
    """ Class to handle the extrinsic camera properties"""

    def r2eul(R) :
        "Convert rotation matrix to euler angles"
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        
        singular = sy < 1e-6
    
        if  not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
        return np.array([x, y, z])
    
    def eul2r(theta) :
        "Convert euler angles to rotation matrix"
        r_x = np.array([[1,                  0,                  0 ],
                        [0, math.cos(theta[0]), -math.sin(theta[0])],
                        [0, math.sin(theta[0]),  math.cos(theta[0])]])
        
        r_y = np.array([[ math.cos(theta[1]), 0, math.sin(theta[1])],
                        [                  0, 1,                 0 ],
                        [-math.sin(theta[1]), 0, math.cos(theta[1])]])

        r_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                        [math.sin(theta[2]),  math.cos(theta[2]), 0],
                        [                 0,                   0, 1]])             
        r = np.dot(r_z, np.dot( r_y, r_x ))
        return r   
    
    def vec2r(vector):
        "Converts rotation vector to rotation matrix"
        if len(vector.shape)>0:
            vector = np.squeeze(vector)
        r = Rotation.from_rotvec(vector).as_matrix()
        return r
    
    def R2vec(R):
        "Converts rotation matrix to rotation vector"
        rotvec = Rotation.from_matrix(R).as_rotvec()
        return rotvec
    
    def cart2pol(x, y):
        "Cartesian to polar coorinates"
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return(rho, phi)
    
    def pol2cart(rho, phi):
        "Polar to cartesian coordinates"
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return(x, y)
    
    def vector_solution(vecA, vecB, vecC, coord):
        A = np.array([ vecA, vecB, vecC])
        A = np.transpose(A)
        b = coord
        X1 = np.dot(np.transpose(A), A)
        X1 = np.linalg.inv(X1)
        X2 = np.dot(np.transpose(A), b)
        machine_xyz = np.dot(X1, X2)
        return machine_xyz
    
#%% Fitting functions

class Fitting:
    """Class for fitting functions"""
    
    def plane_normal(xyz):
        if xyz.shape[1] != 3:
            xyz = np.transpose(xyz)
        A = np.zeros((xyz.shape[0], 3))
        A[:,0] = xyz[:,0]
        A[:,1] = xyz[:,1]
        A[:,2] = np.ones(xyz[:,1].shape)
        b = xyz[:,2]
        X1 = np.dot(np.transpose(A), A)
        X1 = np.linalg.inv(X1)
        X2 = np.dot(np.transpose(A), b)
        X = np.dot(X1, X2)
        res = b - (X[0]*xyz[:,0]) - (X[1]*xyz[:,1]) - X[2]
        n = np.array([-X[0], -X[1], 1])
        n = n / np.sqrt(np.sum(n**2))
        return n, res
    
    def rot_a2b(a, b):
        """ Creates rotation matrix to rotate 3D vector a to 3D vector b """
        # normalise vectors
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        # find rotation axis
        vec = np.cross(a,b)
        vec = vec / np.linalg.norm(vec)
        # find rotation angle
        angle = np.arccos(np.dot(a, b))
        # create rotation matrix
        R = Rotation.from_rotvec(vec*angle).as_matrix()
        return R
    
    def rotate_data_xyz(xyz_data, target_normal = np.array([0, 0, 1])):
        
        """ Rotates xyz_data about the vector of the cross-product of the data
        normal and the target vector by the angle between the vectors """
        
        # Fit plane to data 
        normal, res = Fitting.plane_normal(xyz_data)
        
        # Normalise the vectors
        normal = normal / np.linalg.norm(normal)
        target_normal = target_normal / np.linalg.norm(target_normal)
        
        # Find the axis of rotation
        vec = np.cross(normal,target_normal)
        vec = vec / np.linalg.norm(vec)

        # Find the angle of rotation
        angle = np.arccos(np.dot(normal, target_normal))

        # Rotation matrix
        R = Rotation.from_rotvec(vec*angle).as_matrix()
        
        # Rotate the data
        rotated_data = np.dot(R,xyz_data.transpose()).transpose()
        
        return rotated_data, R
    
    def rotation_matrix_xy(normal, target_normal):
        
        """ Creates rotation matrix to rotate in x and y (around z-axis) 
        about point [0,0] """
        
        # Normalise the vectors
        normal = normal / np.linalg.norm(normal)
        target_normal = target_normal / np.linalg.norm(target_normal)

        # # Find the axis of rotation
        # vec = np.cross(normal,target_normal)
        # vec = vec / np.linalg.norm(vec)

        # Find the angle of rotation
        angle = np.arccos(np.dot(normal, target_normal))
        angle = -angle
        
        # Create matrix values
        xx = np.cos(angle)
        xy = np.sin(angle)
        xz = 0
        
        yx = -np.sin(angle)
        yy = np.cos(angle)
        yz = 0
        
        zx = 0
        zy = 0
        zz = 1
        
        rot_mat = np.array([[xx, yx, zx], 
                            [xy, yy, zy],
                            [xz, yz, zz]])
        
        rot_mat = np.asmatrix(rot_mat)
        return rot_mat
    
    def calc_r(x, y, xc, yc):
        """ calculate the distance of each 2D points 
        from the center (xc, yc) """
        return np.sqrt((x-xc)**2 + (y-yc)**2)
    
    def f(self, c, x, y):
        """ calculate the algebraic distance between 
        the data points and the mean circle centered at c=(xc, yc) """
        Ri = self.calc_r(x, y, *c)
        return Ri - Ri.mean()
    
    def line_fit(self, x, y):
        A = np.array([ x, np.ones(x.shape)])
        if A.shape[0] == 2:
            A = np.transpose(A)
        b = y
        X1 = np.dot(np.transpose(A), A)
        X1 = np.linalg.inv(X1)
        X2 = np.dot(np.transpose(A), b)
        X = np.dot(X1, X2)
        grad = X[0]
        intercept = X[1]
        res = y - (np.dot(A, X))
        return grad, intercept, res
    
    def leastsq_circle(self, x, y):
        # coordinates of the barycenter
        x_m = np.mean(x)
        y_m = np.mean(y)
        center_estimate = x_m, y_m
        center, ier = optimize.leastsq(self.f, 
                                       center_estimate, 
                                       args=(x,y))
        xc, yc = center
        ri       = self.calc_r(x, y, *center)
        r        = ri.mean()
        residu   = (ri - r)
        return xc, yc, r, residu
    
#%% Camera-MachineConversion

def machine2centre(machine, motion_variables):

    # Rotation vector
    angle_machine = (((machine[3] * 
                       np.array(motion_variables['tilt_gradient'])) + 
                      np.array(motion_variables['tilt_offset']) ) *
                     (np.pi/180))
    
    t_x, t_z = Extrinsics.pol2cart(np.array(
                                   motion_variables['tilt_radius']), 
                                   angle_machine)
    
    # Predicted location
    loc = -np.array(motion_variables['xyz_offset']) + \
        (machine[0] * np.array(motion_variables['x-Vector'])) +\
        (machine[1] * np.array(motion_variables['y-Vector'])) +\
        (machine[2] * np.array(motion_variables['z-Vector'])) +\
        np.array([ t_x, 0, t_z])
    
    # Create table rotation
    R_table = Extrinsics.vec2r(
                         np.array(motion_variables['table_vector']) * 
                         machine[4] * 
                         (np.pi/180) * 
                         np.array(motion_variables['table_gradient']))
        
    # Apply table rotation
    loc = np.dot(R_table, loc)
    
    return loc

def machine2rotvec(machine, motion_variables):
    
    # Create table rotation
    angle_table = (machine[4] *
                   np.array(motion_variables['table_gradient']) *
                   (np.pi/180))

    Table_rotation = Rotation.from_rotvec(np.array(motion_variables
                                                   ["R_offset"]
                                                   ["b_axis"]) *
                                          angle_table).as_matrix()
    
    # Create camera tilt rotation
    A_location = machine[3]
    angle_tilt = ((A_location * 
                   np.array(motion_variables['tilt_gradient'])) *
                  (np.pi/180))
    vector_tilt = np.array(motion_variables["R_offset"]["a_axis"])
    
    # Apply rotation of table to camera tilt
    vector_tilt = np.dot(Table_rotation, vector_tilt)
    
    # Create camera rotation
    Tilt_rotation = Rotation.from_rotvec(vector_tilt*angle_tilt).as_matrix()
    
    # Offset vector creation
    offset_vector = np.array([ 
        (motion_variables["R_offset"]["line_x"][0] * angle_tilt) + 
         motion_variables["R_offset"]["line_x"][1],
        (motion_variables["R_offset"]["line_y"][0] * angle_tilt) + 
         motion_variables["R_offset"]["line_y"][1],
        (motion_variables["R_offset"]["line_z"][0] * angle_tilt) + 
         motion_variables["R_offset"]["line_z"][1],] )

    R_offset = Extrinsics.vec2R(offset_vector * 
                                motion_variables["R_offset"]["angle"])
    
    # Generate Camera rotation matrix
    R_camera = np.dot(Table_rotation, np.dot(Tilt_rotation, R_offset))
    
    # Rotation vector
    R_vec = Extrinsics.R2vec(R_camera)
    
    return R_vec

def machine2Camera(machine, motion_variables):
    
    if len(machine.shape) == 1:
        loc = machine2centre(machine, motion_variables)
        R = machine2rotvec(machine, motion_variables)
        T = np.dot(Extrinsics.vec2r(R), loc)
    else:
        loc = []
        R = []
        T = []
        for k in range(machine.shape[0]):
            loc.append(machine2centre(machine[k,:], motion_variables))
            R.append(machine2rotvec(machine[k,:], motion_variables))
            T.append(np.dot(Extrinsics.vec2r(R[k]), -loc[k]))
        loc = np.array(loc) ; R = np.array(R) ; T = np.array(T)
    
    return loc, R, T

def View2machine(camera_view, motion_char):
    
    # Extract number of locations
    if len(camera_view.shape) > 1:
        steps = camera_view.shape[1]
        # Extract conponents
        table_angle = camera_view[3,:]
        elevation = camera_view[4,:]
        centre = camera_view[:3,:]
    else:
        steps = 1
        # Extract conponents
        table_angle = [ camera_view[3] ]
        elevation = [ camera_view[4] ]
        centre = [ camera_view[:3] ]
        centre = np.reshape(centre, (3,1))
    
    # Pre-allocat list of machine locations
    machine = []
    
    for k in range(steps):
        
        # Determine A value
        A_val = (elevation[k] * 
                 motion_char["elevation"][0] + 
                 motion_char["elevation"][1])
        
        machine_a = A_val
        
        # Determine current table machine location
        machine_b = table_angle[k] / motion_char["table_gradient"]
        
        # Apply table rotation to centre
        cur_centre = np.dot(Extrinsics.vec2R(
                                 np.array(motion_char["table_vector"])*
                                 table_angle[k] * (np.pi/180)), 
                            centre[:,k])
        
        # Calculate offset in x and z due to Centre location
        centre_dx = ( np.cos(elevation[k]*(np.pi/180)) * 
                     motion_char["working_distance"])
        centre_dz = (-np.sin(elevation[k]*(np.pi/180)) *
                     motion_char["working_distance"])
        
        # Calculate offset due to camera tilt
        angle_machine = (((A_val * 
                           np.array(motion_char['tilt_gradient'])) + 
                          np.array(motion_char['tilt_offset'])) *
                         (np.pi/180))

        tilt_dx, tilt_dz = Extrinsics.pol2cart(
                           np.array(motion_char['tilt_radius']), 
                           angle_machine)
        
        # Required camera location
        req_pos = cur_centre + np.array([ centre_dx, 0, centre_dz])\
                             + np.array(motion_char["xyz_offset"])\
                             - np.array([ tilt_dx, 0, tilt_dz])
        
        # Find the solution of machine locations
        coord = req_pos
        vec_x = np.array(motion_char["x-Vector"])
        vec_y = np.array(motion_char["y-Vector"])
        vec_z = np.array(motion_char["z-Vector"])
        machine_xyz = Extrinsics.Vector_solution(vec_x, 
                                                 vec_y, 
                                                 vec_z, 
                                                 coord)
        
        # Add machine coordinates
        machine.append([machine_xyz[0], 
                        machine_xyz[1],
                        machine_xyz[2],
                        machine_a,
                        machine_b])
    
    # Convert machine to array
    machine = np.array(machine)
    
    return machine    