import numpy as np
import cv2
import glob
import yaml
from numpy.linalg import inv

"""
    In here is the calibration refinement using bundle adjustment
    Consist of:
        - Reprojection error of Perspective Camera
        - Reprojection error of Microscope 1
        - Reprojection error of Microscope 2
"""

# FOR Calculating RMS Projection Error
def rms_calc(objpoints,rvecs, tvecs, mtx, dist, imgpoints):
    mean_error=0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/ len(imgpoints2)
        mean_error += error
    return(mean_error/len(objpoints))

# FOR Stack the rotational Matrix for Matching the Objectpoint Lengths
def stack_mat_rvecs(a, objpt):
    temp_rvecs = []
    for i in range(len(objpt)):
        temp_rvecs.append(a)

    temp_rvecs = np.stack(temp_rvecs, axis=0)
    return temp_rvecs

# FOR Stack the Matrix for Matching the Objectpoint Lengths
def stack_mat_tvecs(a, objpt):
    temp_tvecs = []
    for i in range(len(objpt)):
        temp_tvecs.append(a)
    
    temp_tvecs = np.stack(temp_tvecs, axis=0)
    return temp_tvecs

# FOR Calculating the Standard Deviation
def stdDeviation_calc(mtx, t1, t2, t3, imgpoints, dist, objpoints):
    # First Calc the transformation
    temp_t1_t2 = np.matmul(t1, t2)
    temp_t1_t2_t3 = np.matmul(temp_t1_t2, t3)

    final_T_rvecs = temp_t1_t2_t3[:3, :3] 
    final_T_tvecs = temp_t1_t2_t3[:3, -1] 
    final_T_tvecs = np.reshape(final_T_tvecs, (3, 1))

    final_T_rvecs = stack_mat_rvecs(final_T_rvecs, imgpoints)
    final_T_tvecs = stack_mat_tvecs(final_T_tvecs, imgpoints)

    for i in range(len(imgpoints)):
        mean_error=0
        imgpoints2, jacobian = cv2.projectPoints(objpoints[i], final_T_rvecs[i], final_T_tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/ len(imgpoints2)
        mean_error += error
    totalDeviation = mean_error/len(objpoints)
    
    return totalDeviation, jacobian



# Load the mic 1 calibration matrix
loadMic1Matrix = yaml.load(open('/home/mscv/Desktop/Internship/latestCalib/params/left_calibration_matrix.yaml'))

# Load the mic 2 calibration matrix
loadMic2Matrix = yaml.load(open('/home/mscv/Desktop/Internship/latestCalib/params/right_calibration_matrix.yaml'))

# Load the mic 1 and mic 2 stereo calibration matrix
loadSterMatrix = yaml.load(open('/home/mscv/Desktop/Internship/latestCalib/params/stereo_calibration_matrix.yaml'))

# Load the Pattern Transformation 
loadPattTrans = yaml.load(open('/home/mscv/Desktop/Internship/latestCalib/params/patternTransformation.yaml'))

# Load the Camera Calibration Matrix
loadCamMatrix = yaml.load(open('/home/mscv/Desktop/Internship/latestCalib/params/cam_calibration_matrix.yaml'))

# Load the set of Homogeneous Transformation
loadHomoTrans = yaml.load(open('/home/mscv/Desktop/Internship/latestCalib/params/homogeneous_transformation_data.yaml'))

# Load objectpoint, imagepoint, rvecs, tvecs, mtx, dist of Perspective Camera
camMatrix = np.asarray(loadCamMatrix['camera_matrix'])
camObj =  np.asarray(loadCamMatrix['object_point'])
camImgpoint = np.asarray(loadCamMatrix['image_point'])
camRvecs = np.asarray(loadCamMatrix['rvecs'])
camTvecs = np.asarray(loadCamMatrix['tvecs'])
camDist = np.asarray(loadCamMatrix['dist'])

# Load objectpoint, imagepoint, rvecs, tvecs, mtx, dist of mic 1 
mic1Matrix = np.asarray(loadMic1Matrix['left_camera_matrix'])
mic1Obj = np.asarray(loadMic1Matrix['left_object_point'])
mic1Imgpoint = np.asarray(loadMic1Matrix['left_image_point'])
mic1Rvecs = np.asarray(loadMic1Matrix['left_rvecs'])
mic1Tvecs = np.asarray(loadMic1Matrix['left_tvecs'])
mic1Dist = np.asarray(loadMic1Matrix['left_dist_coeff'])

# Load objectpoint, imagepoint, rvecs, tvecs, mtx, dist of mic 2
mic2Matrix = np.asarray(loadMic2Matrix['right_camera_matrix'])
mic2Obj = np.asarray(loadMic2Matrix['right_object_point'])
mic2Imgpoint = np.asarray(loadMic2Matrix['right_image_point'])
mic2Dist = np.asarray(loadMic2Matrix['right_dist_coeff'])
mic2Rvecs = np.asarray(loadMic2Matrix['right_rvecs'])
mic2Tvecs = np.asarray(loadMic2Matrix['right_tvecs'])

# Load Homogeneous Transformation of camTM, mic1Tcam, mic2Tcam
camTM = np.asarray(loadHomoTrans['cam_T_M'])
mic1Tcam = np.asarray(loadHomoTrans['mic1_T_cam'])
mic2Tcam = np.asarray(loadHomoTrans['mic2_T_cam'])
MTm = np.asarray(loadHomoTrans['M_T_m'])

# Mic 1 Homogeneous Transformation between mic1Tcam & camTM
mic1_new_homo = np.matmul(mic1Tcam, camTM)

# Mic 2 Homogeneous Transformation between mic2Tcam & camTM
mic2_new_homo = np.matmul(mic2Tcam, camTM)


# Reprojection error of Perspective Camera
"""
    It is the improvement from the general reprojection error by adding homogeneous transformation from pattern and microscope
    reproject_cam_error = rms_calc()
"""

reproject_cam_error = rms_calc(camObj, camRvecs, camTvecs, camMatrix, camDist, camImgpoint)
print("Reprojection Error of Perspective Camera After Bundle Adjustment: ", reproject_cam_error)




# Reprojection error of Mic 1
"""
    It is the improvement from the general reprojection error by adding homogeneous transformation from pattern and microscope
    reproject_mic1_error = rms_calc()
"""
# Get Rvecs and Tvecs from homogeneous transformation matrix of mic1Tcam & camTm
mic1_new_homo_rvecs = mic1_new_homo[:3, :3] 
mic1_new_homo_tvecs = mic1_new_homo[:3, -1] 
mic1_new_homo_tvecs = np.reshape(mic1_new_homo_tvecs, (3, 1))


# For Stacking the Rvecs and Tvecs according to Objectpoint
mic1_rvecs = stack_mat_rvecs(mic1_new_homo_rvecs, mic1Obj)
mic1_tvecs = stack_mat_tvecs(mic1_new_homo_tvecs, mic1Obj)


reproject_mic1_error = rms_calc(mic1Obj, mic1_rvecs, mic1_tvecs, mic1Matrix, mic1Dist, mic1Imgpoint)
print("Reprojection Error of Microscope 1 After Bundle Adjustment: ", reproject_mic1_error)

# Standard Deviation and Imagepoint for Microscope 1
totalDeviation_mic1, jacobian_mic1 = stdDeviation_calc(mic1Matrix, mic1Tcam, camTM, MTm, mic1Imgpoint, mic1Dist, mic1Obj)

print("Standard Deviation average value for Microscope 1: ", totalDeviation_mic1)




# Reprojection error of Mic 2
"""
    It is the improvement from the general reprojection error by adding homogeneous transformation from pattern and microscope
    reproject_mic2_error = rms_calc()
"""

# Get Rvecs and Tvecs from homogeneous transformation matrix of mic2Tcam & camTm
mic2_new_homo_rvecs = mic2_new_homo[:3, :3] 
mic2_new_homo_tvecs = mic2_new_homo[:3, -1] 
mic2_new_homo_tvecs = np.reshape(mic2_new_homo_tvecs, (3, 1))

# For Stacking the Rvecs and Tvecs according to Objectpoint
mic2_rvecs = stack_mat_rvecs(mic2_new_homo_rvecs, mic2Obj)
mic2_tvecs = stack_mat_tvecs(mic2_new_homo_tvecs, mic2Obj)


reproject_mic2_error = rms_calc(mic2Obj, mic2_rvecs, mic2_tvecs, mic2Matrix, mic2Dist, mic2Imgpoint)

print("Reprojection Error of Microscope 2 After Bundle Adjustment: ", reproject_mic2_error)

# Standard Deviation and Imagepoint for Microscope 2
totalDeviation_mic2, jacobian_mic2 = stdDeviation_calc(mic2Matrix, mic2Tcam, camTM, MTm, mic2Imgpoint, mic2Dist, mic2Obj)

print("Standard Deviation average value for Microscope 2: ", totalDeviation_mic2)


# For finding the total reprojection error from the three camera
totalRep_error = reproject_cam_error + reproject_mic1_error + reproject_mic2_error

print("The Total Error for 3 Reprojection Error between Cam, Mic 1, Mic 2 is: ", totalRep_error)




# Reprojection Error to writable lists

reprojectionError_data = { 'reprojectCam_error': np.asarray(reproject_cam_error).tolist(),
                           'reprojectMic1_error': np.asarray(reproject_mic1_error).tolist(),
                           'reprojectMic2_error': np.asarray(reproject_mic2_error).tolist(),
                           'mic1_new_homogeneous': np.asarray(mic1_new_homo).tolist(),
                           'mic2_new_homogeneous': np.asarray(mic2_new_homo).tolist(),
                           'total_reprojection_error': np.asarray(totalRep_error).tolist(),
                           'standardDeviation_mic1': np.asarray(totalDeviation_mic1).tolist(),
                           'standardDeviation_mic2': np.asarray(totalDeviation_mic2).tolist()}

# and save it to a file
with open("/home/mscv/Desktop/Internship/latestCalib/params/reprojectionErrorParam.yaml", "w") as f:
    yaml.dump(reprojectionError_data, f)


