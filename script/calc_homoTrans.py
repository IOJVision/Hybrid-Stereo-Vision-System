import numpy as np
import cv2
import glob
import yaml
from numpy.linalg import inv

"""
    In Here is the calculation for the Homogeneous Transformation of:
        - Small Pattern to Big Pattern, mTM
            - Big Pattern to Small Pattern, MTm
        - Microscope 1 to Camera, mic1Tcam
            - Camera to Microscope 1, camTmic1
        - Microscope 2 to Camera, mic2Tcam
            - Camera to Microscope 2, camTmic2
        - Microscope 1 to Microscope 2, mic1Tmic2
"""

# homogenTrans function is for create 4 by 4 Matrix consist of rotation matrix and translation matrix
def homogenTrans(R, T):
    matrix_Homo = np.zeros((4,4))
    matrix_Homo[:3, :3] = R
    matrix_Homo[:3, -1]= np.squeeze(T)
    matrix_Homo[3, 3]= 1
    return matrix_Homo

# Homogeneous Transformation Matrix for mic 1 and camera
def calc_mic1Tcam(camR, camT, mic1R, mic1T, mTM):
    """ Required:
        - Homogeneous Tranformation of Camera and Big Pattern
        - Homogeneous Tranformation of Small Pattern to Big Pattern, mTM
        - Homogeneous Tranformation of mic 1 and small pattern
    """
    # Homo Trans cam & big patt
    camTM = homogenTrans(camR, camT)
    # Homo Trans mic1 & Small patt
    micTm = homogenTrans(mic1R, mic1T)

    #Homo Trans mic 1 & cam
    temp_mic1Tcam = np.matmul(camTM, mTM)
    final_mic1Tcam = np.matmul(temp_mic1Tcam, inv(micTm))
    return final_mic1Tcam

# Homogeneous Transformation Matrix for mic 2 and camera
def calc_mic2Tcam(camR, camT, mic1R, mic1T, mTM):
    """ Required:
        - Homogeneous Tranformation of Camera and Big Pattern
        - Homogeneous Tranformation of Small Pattern to Big Pattern, mTM
        - Homogeneous Tranformation of mic 2 and small pattern
    """
    # Homo Trans cam & big patt
    camTM = homogenTrans(camR, camT)
    # Homo Trans mic 2 & Small patt
    micTm = homogenTrans(mic2R, mic2T)

    #Homo Trans mic 2 & cam
    temp_mic2Tcam = np.matmul(camTM, mTM)
    final_mic2Tcam = np.matmul(temp_mic2Tcam, inv(micTm))
    return final_mic2Tcam

# Homogeneous Transformation Matrix for Mic and small pattern
def calc_micTpatt(rvecs, tvecs):
    micTm = homogenTrans(rvecs, tvecs)
    return micTm


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

# Rotation Matrix and Translation Matrix for mic 1 and mic 2 (stereo)
micR =  np.array(loadSterMatrix['rotation_matrix'])
micT = np.array(loadSterMatrix['translation_vector'])

# Rotation Matrix and Translation Matrix for cam and big pattern
camR =  np.array(loadCamMatrix['rvec_mat'])
camT = np.array(loadCamMatrix['tvec_mat'])

# Rotation Matrix and Translation Matrix for mic 1 and small pattern
mic1R =  np.array(loadSterMatrix['left_rotation_matrix'])
mic1T = np.array(loadMic1Matrix['left_translation_mat'])

# Rotation Matrix and Translation Matrix for mic 2 and small pattern
mic2R =  np.array(loadSterMatrix['right_rotation_matrix'])
mic2T = np.array(loadMic2Matrix['right_translation_mat'])

# Homogeneous Transformation Matrix for Pattern
# In pixel unit
mTMinPixel = np.asarray(loadPattTrans['mTM'])
MTminPixel = np.asarray(loadPattTrans['MTm'])

# In mm unit
#mTM = np.asarray(loadPattTrans['mTMinMM'])
#MTm = np.asarray(loadPattTrans['MTminMM'])
mTM = np.asarray(loadPattTrans['mTM'])
MTm = np.asarray(loadPattTrans['MTm'])

# Transformation of Mic 1 and camera
mic1Tcam = calc_mic1Tcam(camR, camT, mic1R, mic1T, mTM)
print("Homogeneous Transformation of Microscope 1 and Camera: \n", mic1Tcam)

# Transformation of camera and Mic 1
camTmic1 = inv(mic1Tcam)
print("Homogeneous Transformation of Camera and Microscope 1: \n", camTmic1)

# Transformation of Mic 2 and camera
mic2Tcam = calc_mic2Tcam(camR, camT, mic2R, mic2T, mTM)
print("Homogeneous Transformation of Microscope 2 and Camera: \n", mic2Tcam)

# Transformation of camera and Mic 2
camTmic2 = inv(mic2Tcam)
print("Homogeneous Transformation of Camera and Microscope 2: \n", camTmic2)

# Transformation of mic 1 and mic 2
mic1Tmic2 = homogenTrans(micR, micT)
print("Homogeneous Transformation of Microscope 1 and Microscope 2: \n", mic1Tmic2)

# Transformation of cam and big pattern
cam_T_M = homogenTrans(camR, camT)
print("Homogeneous Transformation of Camera and big pattern: \n", cam_T_M)

# Transformation of mic 1 and small pattern
mic1_T_m = calc_micTpatt(mic1R, mic1T)
print("Homogeneous Transformation of Microscope 1 and small pattern: \n", mic1_T_m)

# Transformation of mic 1 and small pattern
mic2_T_m = calc_micTpatt(mic2R, mic2T)
print("Homogeneous Transformation of Microscope 2 and small pattern: \n", mic2_T_m)

# transform the matrix and distortion coefficients to writable lists
transformation_data = { 'm_T_M': np.asarray(mTM).tolist(),
                        'M_T_m': np.asarray(MTm).tolist(),
                        'mic1_T_cam': np.asarray(mic1Tcam).tolist(),
                        'cam_T_mic1': np.asarray(camTmic1).tolist(),
                        'mic2_T_cam': np.asarray(mic2Tcam).tolist(),
                        'cam_T_mic2': np.asarray(camTmic2).tolist(),
                        'mic1_T_mic2': np.asarray(mic1Tmic2).tolist(),
                        'cam_T_M':np.asarray(cam_T_M).tolist(),
                        'mic1_T_m':np.asarray(mic1_T_m).tolist(),
                        'mic2_T_m': np.asarray(mic2_T_m).tolist()}

# and save it to a file
with open("/home/mscv/Desktop/Internship/latestCalib/params/homogeneous_transformation_data.yaml", "w") as f:
    yaml.dump(transformation_data, f)








