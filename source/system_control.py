# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 11:17:55 2022

@author: Emma Woods

This module is used to connect to the system, namely the cameras and the duet 
board (which controls the lights and motion stages). Further additions would
include connecting to the projector as well. 

The duet board is controlled using g-code for which a library of commands can
be found at https://docs.duet3d.com/User_manual/Reference/Gcodes although all
the main ones will already ave been implemented. 

The cameras are basler cameras which are controled using the pypylon module,
and can also be controlled manually in the Pylon Viewer software which should
be installed already on the P2 computer. 


"""
## External Libraries
# needed for duet board control
import serial
import serial.tools.list_ports
import time
from time import sleep
import json

# needed for camera control
from pypylon import pylon
import cv2 as cv
import numpy as np


#%% Open / Close Connections to Hardware

def initialise_connection(cameras_used, exposure, port):
    """
    Initialise connections to the camera(s) and duet board

    Parameters
    ----------
    cameras_used : int
        Number of cameras to connect to
    exposure : list
        List of exposure values for each camera.
    port : str
        com port string for duet board - obtained from sc.ListPorts

    Returns
    -------
    board : TYPE
        DESCRIPTION.
    camera : TYPE
        DESCRIPTION.

    """
    
    # Connect to the camera
    camera = Camera.connect(cameras_used, exposure)
    
    # Connect to smoothie
    board = DuetBoard.connect(port)
    
    DuetBoard.lights(board, True)
    # Home board
    DuetBoard.home(board)
    
    return board, camera

def close_connections(board, camera):
    """ Closes the connections to the camera(s) and duet board. """
    DuetBoard.lights(board, False)
    board.close()  
    camera.Close()

#%%  Automatically extract the port name

def list_ports():
    """
    Returns
    -------
    port : STR
        name of the COM port Duet Board is conencted to.

    """
    ports = serial.tools.list_ports.comports()
    ports_list = []
    for port, desc, _hwid in sorted(ports):
        ports_list.append(("{}: {}".format(port, desc), port))

    if len(ports_list) == 0:
        raise UnboundLocalError
        # ports_list.append(('No devices detected on com ports!'))
    return port

#%%
# board = DuetBoard(port)
 # board.move((x,y,z))
class DuetBoard:
    """ Class for all duet board controls """
            
    def connect(port):
        """Connect to the board"""
        # Settings
        baud = 115200
        
        # Connect to board
        board = serial.Serial(port, baud, timeout=.1)
        
        # Confirm absolute coordinate system
        board.read(10000)
        board.write('G90 \n'.encode('utf-8'))
        return board
    
    def move(Location, board):
        """ Move to Location """
        # Create location string
        wr_str = 'G0 X' + str(Location[0]) + \
                   ' Y' + str(Location[1]) + \
                   ' Z' + str(Location[2]) + \
                   ' A' + str(Location[3]) + \
                   ' B' + str(Location[4])
            
        # Write to board    
        board.write(wr_str.encode('utf-8'))
        board.read(10000)
        
        sleep(0.1)
        
        try:
            status = DuetBoard.get_status(board)
        except:
            return status
        
        while DuetBoard.get_status(board)!="I":
            time.sleep(0.1)
        time.sleep(0.5)
        
                
    def lights(board, state):
        """ Turn lights on or off """
        if state:
            board.write('M80 \n'.encode('utf-8'))
            board.read(10000)
        else:
            board.write('M81 \n'.encode('utf-8'))
            board.read(10000)
        
    def get_status(board):
        board.read(10000)
        board.write('M408 \n'.encode('utf-8'))
        string = board.read(10000)
        try:
            if len(string)>10:
                obj = json.loads(string[:-5])
                return obj["status"]
            else:
                return "Error"
        except:
            return string
    
        
    def home(board):
        """ Home movement stage"""
        # Write to board
        board.read(10000)
        board.write('G28 \n'.encode('utf-8'))
        
        while board.read(100) != b'ok\n':
            time.sleep(0.1)

#%% Camera functions

class Camera:
    """Class for handling camera control"""
    
    def connect(maxCamerasToUse, Exposure_time):
        """ Connect to the camera(s)"""
        # Get the transport layer factory.
        tlFactory = pylon.TlFactory.GetInstance()
        
        # Get all attached devices and exit application if no device is found.
        devices = tlFactory.EnumerateDevices()
        if len(devices) == 0:
            raise pylon.RuntimeException("No camera present.")
        
        # Create an array of instant cameras for the found devices and 
        # avoid exceeding a maximum number of devices.
        cameras = pylon.InstantCameraArray(min(len(devices), maxCamerasToUse))
        
        # Create and attach all Pylon Devices.
        for i, cam in enumerate(cameras):
            cam.Attach(tlFactory.CreateDevice(devices[i]))
        
            # Print the model name of the camera.
            print("Using device ", cam.GetDeviceInfo().GetModelName())
        
        # Connect to camera and set to hardware trigger
        cameras.Open()
        for i, camera in enumerate(cameras):
            
            camera.AcquisitionMode.SetValue('Continuous')
            camera.MaxNumBuffer = 15
            
            # Set exposure time to 1/60th of a second
            camera.ExposureTime.SetValue(Exposure_time[i])
        
        return cameras
        
    def camera_capture(cameras):
        """ Capture images """
        # Pre-allcate image array
        Images = []
        
        # Starts grabbing for all cameras
        cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        
        # Capture from all cameras
        for k in range(cameras.GetSize()):
            
            grabResult = cameras[k].RetrieveResult(500, 
                             pylon.TimeoutHandling_ThrowException)
            
            # Get image set and assign to Images
            Images.append(grabResult.GetArray())
        
        # Stop camera grabbing
        cameras.StopGrabbing()
        
        if len(Images)==1:
            Images = Images[0]
        
        return Images
    
    def live_view(cameras):
        """ Create live view window """    
        # Create window
        cv.namedWindow('Acquisition', cv.WINDOW_NORMAL)
        cv.resizeWindow('Acquisition', 500, 500)
        
        # Starts grabbing for all cameras
        cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        
        while cameras[0].IsGrabbing():
            
            # Capture image from first camera
            grabResult = cameras[0].RetrieveResult(500000, 
                             pylon.TimeoutHandling_ThrowException)
            Images = grabResult.GetArray()
            
            # Get the rest of the cameras
            for k in range(1,cameras.GetSize()):
                
                grabResult = cameras[k].RetrieveResult(500000, 
                                 pylon.TimeoutHandling_ThrowException)
                im = grabResult.GetArray()
            
                Images = np.concatenate((Images, im), axis=1)
            
            # If ESC is pressed exit and destroy window
            cv.imshow('Acquisition', Images)
            
            if cv.waitKey(100) & 0xFF == 27:
                break
        
        cameras.StopGrabbing()
        
        cv.destroyAllWindows()
