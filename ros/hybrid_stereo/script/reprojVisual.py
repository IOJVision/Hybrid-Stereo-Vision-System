#!/usr/bin/env python

import rospy
import numpy as np
import cv2
import message_filters
from cv_bridge import CvBridge
from std_msgs.msg import UInt8, Float64
from sensor_msgs.msg import Image, CompressedImage
import yaml
import open3d as o3d
from open3d import *
from numpy import random as rnd
import sys
from numpy.linalg import inv
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Import the necessary stuff
# Load the optimization data
loadDataOp = yaml.load(open('/home/mscv/Desktop/Internship/1705_circleCalib/params/data_op.yaml'))

# Load Camera calib
loadCamCalib = yaml.load(open('/home/mscv/Desktop/Internship/1705_circleCalib/params/cam_calib.yaml'))

# Load Mic 1 calib
loadMic1Calib = yaml.load(open('/home/mscv/Desktop/Internship/1705_circleCalib/params/left_calib.yaml'))

# Load Mic 2 calib
loadMic2Calib = yaml.load(open('/home/mscv/Desktop/Internship/1705_circleCalib/params/right_calib.yaml'))

# Load the mic 1 and mic 2 stereo calibration matrix
loadStereoCalib = yaml.load(open('/home/mscv/Desktop/Internship/1705_circleCalib/params/stereo_calibration_matrix.yaml'))

# Load the Pattern Transformation 
loadPattTrans = yaml.load(open('/home/mscv/Desktop/Internship/1705_circleCalib/params/patternTransform.yaml'))

# Load The Optimize Mtx
camMtx = np.asarray(loadDataOp['K_cam'])
mic1Mtx = np.asarray(loadDataOp['K_mic1'])
mic2Mtx = np.asarray(loadDataOp['K_mic2'])

# Load The objp and distortion
camDist = np.asarray(loadCamCalib['dist_coeff'])
mic1Dist = np.asarray(loadMic1Calib['left_dist_coeff'])
mic2Dist = np.asarray(loadMic2Calib['right_dist_coeff'])
mic1Obj = np.asarray(loadMic1Calib['objpoint'])
mic2Obj = np.asarray(loadMic2Calib['objpoint'])

# Load Homogeneous Transformation of mTM, mic1Tcam, mic2Tcam
mic1Tcam = np.asarray(loadDataOp['mic1Tcam'])
mic2Tcam = np.asarray(loadDataOp['mic2Tcam'])
MTm = np.asarray(loadDataOp['MTm'])
mTM = np.asarray(loadDataOp['mTM'])

# Load Rectification of Stereo
K1 = np.asarray(loadStereoCalib['left_camera_matrix'])
D1 = np.asarray(loadStereoCalib['left_dist_coeff'])
K2 = np.asarray(loadStereoCalib['right_camera_matrix'])
D2 = np.asarray(loadStereoCalib['right_dist_coeff'])
R = np.asarray(loadStereoCalib['rotation_matrix'])
T = np.asarray(loadStereoCalib['translation_vector'])
E = np.asarray(loadStereoCalib['essential_matrix'])
F = np.asarray(loadStereoCalib['fundamental_matrix'])
R1 = np.asarray(loadStereoCalib['left_rotation_matrix'])
R2 = np.asarray(loadStereoCalib['right_rotation_matrix'])
P1 = np.asarray(loadStereoCalib['left_projection_coordinate_matrix'])
P2 = np.asarray(loadStereoCalib['right_projection_coordinate_matrix'])
Q = np.asarray(loadStereoCalib['disparity_to_depth_mapping_matrix'])
roi_left = np.asarray(loadStereoCalib['roi_left'])
roi_right = np.asarray(loadStereoCalib['roi_right'])

# Declare Publisher for CAM
cam_Reproj = rospy.Publisher('/hybrid_vision/cam_reproj', Image, queue_size=1)

# Declare Publisher for mic 1
mic1_Reproj = rospy.Publisher('/hybrid_vision/mci1_reproj', Image, queue_size=1)

# Declare Publisher for mic 2
mic2_Reproj = rospy.Publisher('/hybrid_vision/mci2_reproj', Image, queue_size=1)

# Declare Publisher for Disparity with Target Point
disp_Reproj = rospy.Publisher('/hybrid_vision/disp_reproj', Image, queue_size=1)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

########################################Blob Detector##############################################

# Setup SimpleBlobDetector parameters.
blobParams = cv2.SimpleBlobDetector_Params()

# Change thresholds
blobParams.minThreshold = 0
blobParams.maxThreshold = 255

# Filter by Area.
blobParams.filterByArea = True
blobParams.minArea = 1     # minArea may be adjusted to suit for your experiment
blobParams.maxArea = 2500   # maxArea may be adjusted to suit for your experiment

# Filter by Circularity
blobParams.filterByCircularity = True
blobParams.minCircularity = 0.1

# Filter by Convexity
blobParams.filterByConvexity = True
blobParams.minConvexity = 0.87

# Filter by Inertia
blobParams.filterByInertia = True
blobParams.minInertiaRatio = 0.01

# Create a detector with the parameters
blobDetector = cv2.SimpleBlobDetector_create(blobParams)

#row, original 6 but put 5, a
#column, original 7 but put 6, b
a = 6
b = 7

# Objectpoint for camera
objp = np.zeros((a*b,3), np.float32)
objp[:,:2] = np.mgrid[0:b,0:a].T.reshape(-1,2)*20

# Objectpoint for mic1 and mic2
objpMic = np.zeros((a*b,3), np.float32)
objpMic[:,:2] = np.mgrid[0:b,0:a].T.reshape(-1,2)*5

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
left_imgpoints = [] # 2d points in image plane.
right_imgpoints = [] # 2d points in image plane.

# FOR Calculating Homogeneous Transformation
def homogenTrans(R, T):
    matrix_Homo_final = []
    for i in range(len(R)):
        matrix_Homo = np.zeros((4,4))
        matrix_Homo[:3, :3] = R
        matrix_Homo[:3, -1]= np.squeeze(T)
        matrix_Homo[3, 3]= 1
        #matrix_Homo_final.append(matrix_Homo)
    #matrix_Homo_final = np.stack(matrix_Homo_final, axis=0)
    return matrix_Homo

# FOR Calculating RMS Projection Error
def rms_calc(objpoints,rvecs, tvecs, mtx, dist):
    mean_error=0
    imgpoints2, jacob = cv2.projectPoints(objpoints[0], rvecs, tvecs, mtx, dist)
    
    return imgpoints2

def calc_depth_map(imgL, imgR):
    """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map ( left to right disparity ) """
    # SGBM Parameters -----------------
    window_size = 5  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-1,
        numDisparities=5*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=8 * 3 * window_size,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size,
        disp12MaxDiff=12,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 80000
    sigma = 1.3
    visual_multiplier = 6

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)

    return filteredImg

def calc_xm1(mic1Tcam, camTM, Mtm):
    # A = mic1Tcam * camTM
    temp_xm1_mic1 = np.matmul(mic1Tcam, camTM)


    # B = A * MTm
    temp2_xm1_mic1 = np.matmul(temp_xm1_mic1, MTm)

    # Getting Rvecs and Tvecs from xm1
    temp_xm1_rvecs = temp2_xm1_mic1[:3, :3]
    temp_xm1_tvecs = temp2_xm1_mic1[:3, -1]
    
    temp2_xm1_tvecs = np.reshape(temp_xm1_tvecs, (3,1))


    return temp_xm1_rvecs, temp2_xm1_tvecs

def calc_camTM(camRvecs, camTvecs):
    # Perspective Camera Rvecs to Matrix
    cam_rmat = []
    temp_cam_rmat = []

    for i in range(len(camRvecs)):
        temp_cam_rmat, jacobian_cam = cv2.Rodrigues(camRvecs)
        cam_rmat.append(temp_cam_rmat)
    #cam_rmat = np.stack(cam_rmat, axis=0)

    matrix_Homo = np.zeros((4,4))
    matrix_Homo[:3, :3] = temp_cam_rmat
    matrix_Homo[:3, -1]= np.squeeze(camTvecs)
    matrix_Homo[3, 3]= 1
    camTM = matrix_Homo

    return camTM



def reproj_vision(cam, mic1, mic2):
    bridge = CvBridge()

    cam_img  = bridge.imgmsg_to_cv2(cam, "bgr8")
    mic1_img = bridge.imgmsg_to_cv2(mic1, "bgr8")
    mic2_img = bridge.imgmsg_to_cv2(mic2, "bgr8")

    
    gray = cv2.cvtColor(cam_img,cv2.COLOR_BGR2GRAY)

    # Detect blobs
    keypoints = blobDetector.detect(gray)

    # Draw detected blobs as red circles. This helps cv2.findCirclesGrid() .
    im_with_keypoints = cv2.drawKeypoints(cam_img, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    im_with_keypoints_gray = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findCirclesGrid(im_with_keypoints_gray, (b, a), None, flags = cv2.CALIB_CB_SYMMETRIC_GRID)   # Find the circle grid

    if ret == True:
        # Certainly, every loop objp is the same, in 3D.
        objpoints.append(objp)
        # Refines the corner locations.
        corners2 = cv2.cornerSubPix(im_with_keypoints_gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Find the rotation and translation vectors.
        _,rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, camMtx, camDist)
        camRvecs=rvecs
        camTvecs=tvecs
    else:
        rospy.loginfo("Cannot Detect The object, Please Clear Up the Perspective Camera View")
        rospy.sleep(5)




    # Right object point
    gray_right = cv2.cvtColor(mic2_img, cv2.COLOR_BGR2GRAY)

    # Detect blobs Right
    right_keypoints = blobDetector.detect(gray_right)

    # Draw detected blobs as red circles Right. This helps cv2.findCirclesGrid().
    right_im_with_keypoints = cv2.drawKeypoints(mic2_img, right_keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    right_im_with_keypoints_gray = cv2.cvtColor(right_im_with_keypoints, cv2.COLOR_BGR2GRAY)
    right_ret, right_corners = cv2.findCirclesGrid(right_im_with_keypoints, (b, a), None, flags = cv2.CALIB_CB_SYMMETRIC_GRID)   # Find the circle grid

    # Left object point
    gray_left = cv2.cvtColor(mic1_img, cv2.COLOR_BGR2GRAY)

    # Detect blobs Left
    left_keypoints = blobDetector.detect(gray_left)

    # Draw detected blobs as red circles Left. This helps cv2.findCirclesGrid().
    left_im_with_keypoints = cv2.drawKeypoints(mic1_img, left_keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    left_im_with_keypoints_gray = cv2.cvtColor(left_im_with_keypoints, cv2.COLOR_BGR2GRAY)
    left_ret, left_corners = cv2.findCirclesGrid(left_im_with_keypoints, (b, a), None, flags = cv2.CALIB_CB_SYMMETRIC_GRID)   # Find the circle grid

    if right_ret and left_ret == True:  # If both image is okay. 
        # Certainly, every loop objp is the same, in 3D.
        objpoints.append(objp)

        # Refines the corner locations Right.
        corners2_right = cv2.cornerSubPix(right_im_with_keypoints_gray,right_corners,(11,11),(-1,-1),criteria)
        right_imgpoints.append(corners2_right)

        # Find the rotation and translation vectors.
        _,right_rvecs, right_tvecs, right_inliers = cv2.solvePnPRansac(objpMic, corners2_right, mic2Mtx, mic2Dist)
        mic2Rvecs=right_rvecs
        mic2Tvecs=right_tvecs

        # Refines the corner locations Left.
        corners2_left = cv2.cornerSubPix(left_im_with_keypoints_gray,left_corners,(11,11),(-1,-1),criteria)
        left_imgpoints.append(corners2_left)

        # Find the rotation and translation vectors.
        _,left_rvecs, left_tvecs, left_inliers = cv2.solvePnPRansac(objpMic, corners2_left, mic1Mtx, mic1Dist)
        mic1Rvecs=left_rvecs
        mic1Tvecs=left_tvecs
        
    rospy.loginfo("Detecting the object in Perspective Camera")
    #rospy.sleep(5)
    # Calculate camTM
    rospy.loginfo("Calculating the Rotation and Translation of the Detected Point")
    camTM = calc_camTM(camRvecs, camTvecs)
    #rospy.sleep(5)
    rospy.loginfo("Calculating the new projection from computed camTM")
    #rospy.sleep(5)
    # Calculate the Reprojection for mic1 from extracted point in cam 
    # xm1
    xm1_rvecs, xm1_tvecs = calc_xm1(mic1Tcam, camTM, MTm)
    
    # Getting Projection Point from xm1
    imgpts_xm1 = rms_calc(mic1Obj, xm1_rvecs, xm1_tvecs, mic1Mtx, mic1Dist)
    
    # Draw the circle at detected point
    imgpts_xm1Int = np.int0(imgpts_xm1)

    rospy.loginfo("Done Calculate XM1")

    # we iterate through each corner,  
    # making a circle at each point that we think is a corner. 
    for i in imgpts_xm1Int: 
        x, y = i.ravel() 
        cv2.circle(mic1_img, (x, y), 3, 255, -1) 

    reprojectMic1 = mic1_img
    mic1Rep(reprojectMic1)

    
    
    # Calculate the Reprojection for mic1 from extracted point in cam 
    # xm1
    xm2_rvecs, xm2_tvecs = calc_xm1(mic2Tcam, camTM, MTm)
    
    # Getting Projection Point from xm1
    imgpts_xm2 = rms_calc(mic2Obj, xm2_rvecs, xm2_tvecs, mic2Mtx, mic2Dist)
    
    # Draw the circle at detected point
    imgpts_xm2Int = np.int0(imgpts_xm2)

    rospy.loginfo("Done Calculate XM2")
    # we iterate through each corner,  
    # making a circle at each point that we think is a corner. 
    for i in imgpts_xm2Int: 
        x, y = i.ravel() 
        cv2.circle(mic2_img, (x, y), 3, 255, -1) 

    reprojectMic2 = mic2_img
    mic2Rep(reprojectMic2)

    # For camera 

    # Draw the circle at detected point
    imgpts_cam = np.int0(corners2)

    # we iterate through each corner,  
    # making a circle at each point that we think is a corner. 
    for i in imgpts_cam: 
        x, y = i.ravel() 
        cv2.circle(cam_img, (x, y), 3, 255, -1) 
    
    reprojectCam = cam_img
    camRep(reprojectCam)

    # Get disparity
    h, w = mic1_img.shape[:2]

    #Get optimal camera matrix for better undistortion 
    new_camera_matrix1, roi1 = cv2.getOptimalNewCameraMatrix(K1,D1,(w,h),1,(w,h))

    #Get optimal camera matrix for better undistortion 
    new_camera_matrix2, roi2 = cv2.getOptimalNewCameraMatrix(K2,D2,(w,h),1,(w,h))

    #Undistort images
    img_1_undistorted = cv2.undistort(mic1_img, K1, D1, None, new_camera_matrix1)
    img_2_undistorted = cv2.undistort(mic2_img, K2, D2, None, new_camera_matrix2)

    #Disparity Map
    newdisp = calc_depth_map(img_1_undistorted, img_2_undistorted)
    projImg(newdisp)

def projImg(dispImg):
    bridge = CvBridge()
    disp_Reproj.publish(bridge.cv2_to_imgmsg(dispImg))

def mic1Rep(mic1Img):
    bridge = CvBridge()
    mic1_Reproj.publish(bridge.cv2_to_imgmsg(mic1Img, encoding="rgb8"))

def mic2Rep(mic2Img):
    bridge = CvBridge()
    mic2_Reproj.publish(bridge.cv2_to_imgmsg(mic2Img, encoding="rgb8"))
    
def camRep(camImg):
    bridge = CvBridge()
    cam_Reproj.publish(bridge.cv2_to_imgmsg(camImg, encoding="rgb8"))


def listener():
    rospy.init_node('reproj_vision', anonymous=True)
    cam = message_filters.Subscriber('/camera_01/image_raw', Image)
    mic1 = message_filters.Subscriber('/camera_02/image_raw', Image)
    mic2 = message_filters.Subscriber('/camera_00/image_raw', Image)
    ts = message_filters.ApproximateTimeSynchronizer([cam, mic1, mic2], 10, 5)
    ts.registerCallback(reproj_vision)
    rospy.spin()

if __name__ == '__main__':
    listener()

