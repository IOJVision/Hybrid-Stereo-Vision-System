import numpy as np 
import cv2
import glob
import sys
import yaml

# For stereo calibration

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
image_size = None

# Standardized Checkerboard Size
a = 6 #row, original 7 but put 6, a
b = 8 #column, original 9 but put 8, b

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((a*b,3), np.float32)
objp[:,:2] = np.mgrid[0:b,0:a].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
left_imgpoints = [] # 2d points in image plane.
right_imgpoints = [] # 2d points in image plane.

# Load the image for left and right
left_images = glob.glob('/home/mscv/Desktop/Internship/latestCalib/data/left/*.jpg') # Mic 1
right_images = glob.glob('/home/mscv/Desktop/Internship/latestCalib/data/right/*.jpg') # Mic 2

# Images should be perfect pairs. Otherwise all the calibration will be false.
# Be sure that first cam and second cam images are correctly prefixed and numbers are ordered as pairs.
# Sort will fix the globs to make sure.
left_images.sort()
right_images.sort()

# Pairs should be same size
if len(left_images) != len(right_images):
    print("Numbers of left and right images are not equal. They should be pairs.")
    print("Left images count: ", len(left_images))
    print("Right images count: ", len(right_images))
    sys.exit(-1)

pair_images = zip(left_images, right_images) # Pair the images for single loop handling

# FOR Calculating RMS Projection Error
def rms_calc(objpoints,rvecs, tvecs, mtx, dist, imgpoints):
    mean_error=0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/ len(imgpoints2)
        mean_error += error
    return(mean_error/len(objpoints))

# FOR Calculating Homogeneous Transformation
def homogenTrans(R, T):
    matrix_Homo = np.zeros((4,4))
    matrix_Homo[:3, :3] = R
    matrix_Homo[:3, -1]= np.squeeze(T)
    matrix_Homo[3, 3]= 1
    return matrix_Homo

# Iterate throught the pairs and find chessboard corner. Add them to arrays.\
# If the corner not found, remove a pair
for left_im, right_im in pair_images:
    # Right object point
    right = cv2.imread(right_im)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners for right
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, (b,a), None)
    print("Right: ", ret_right)

    # Left object point
    left = cv2.imread(left_im)
    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners for left
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, (b,a), None)
    print("Left: ", ret_left)

    if ret_left and ret_right: # If both image is okay. 
        # Object points
        objpoints.append(objp)

        # Right points
        corners2_right = cv2.cornerSubPix(gray_right, corners_right, (11,11), (-1,-1), criteria)
        right_imgpoints.append(corners2_right)

        # Draw and display the corners of Right Image
        right = cv2.drawChessboardCorners(right, (b,a), corners2_right, ret_right)
        #cv2.imshow('Right_img',right)
        #cv2.waitKey(5000)

        # Left points
        corners2_left = cv2.cornerSubPix(gray_left, corners_left, (11,11), (-1,-1), criteria)
        left_imgpoints.append(corners2_left)

        # Draw and display the corners of Left Image
        left = cv2.drawChessboardCorners(left, (b,a), corners2_left, ret_left)
        #cv2.imshow('Left_img',left)
        #cv2.waitKey(5000)
    else:
        print("Chessboard couldn't detected. Image pair: ", left_im, " and ", right_im)
        continue

cv2.destroyAllWindows()
# Left Calibration
ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(objpoints, left_imgpoints, gray_left.shape[::-1],None,None)

# solvePnP to get the rvec matrix and tvec matrix
left_retval, left_rvec, left_tvec = cv2.solvePnP(objp, corners2_left, mtx_left, dist_left)

# Left Re-projection error gives a good estimation of just how exact the found parameters are
left_total_error = rms_calc(objpoints,rvecs_left, tvecs_left, mtx_left, dist_left, left_imgpoints)
print(left_total_error)

# transform the matrix and distortion coefficients to writable lists
left_data = {'left_ret': np.asarray(ret_left).tolist(),
            'left_camera_matrix': np.asarray(mtx_left).tolist(),
            'left_dist_coeff': np.asarray(dist_left).tolist(),
            'left_rvecs': np.asarray(rvecs_left).tolist(),
            'left_tvecs': np.asarray(tvecs_left).tolist(),
            'left_total_error': np.asarray(left_total_error).tolist(),
            'left_translation_mat': np.asarray(left_tvec).tolist(),
            'left_object_point': np.asarray(objpoints).tolist(),
            'left_image_point': np.asarray(left_imgpoints).tolist()}

# and save it to a file
with open("/home/mscv/Desktop/Internship/latestCalib/params/left_calibration_matrix.yaml", "w") as f:
    yaml.dump(left_data, f)


# Right Calibration
ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(objpoints, right_imgpoints, gray_right.shape[::-1],None,None)

# solvePnP to get the rvec matrix and tvec matrix
right_retval, right_rvec, right_tvec = cv2.solvePnP(objp, corners2_right, mtx_right, dist_right)

# Right Re-projection error gives a good estimation of just how exact the found parameters are
right_total_error = rms_calc(objpoints, rvecs_right, tvecs_right, mtx_right, dist_right, right_imgpoints)
print(right_total_error)

# transform the matrix and distortion coefficients to writable lists
right_data = {  'right_ret': np.asarray(ret_right).tolist(),
                'right_camera_matrix': np.asarray(mtx_right).tolist(),
                'right_dist_coeff': np.asarray(dist_right).tolist(),
                'right_rvecs': np.asarray(rvecs_right).tolist(),
                'right_tvecs': np.asarray(tvecs_right).tolist(),
                'right_total_error':np.asarray(right_total_error).tolist(),
                'right_translation_mat': np.asarray(right_tvec).tolist(),
                'right_object_point': np.asarray(objpoints).tolist(),
                'right_image_point': np.asarray(right_imgpoints).tolist()}

# and save it to a file
with open("/home/mscv/Desktop/Internship/latestCalib/params/right_calibration_matrix.yaml", "w") as f:
    yaml.dump(right_data, f)


# Stereo Calibration
stereo_ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(objpoints, left_imgpoints, right_imgpoints, mtx_left, dist_left, mtx_right, dist_right, None)

# Stereo Rectify, Computes rectification transforms for each head of a calibrated stereo camera.
R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(K1, D1, K2, D2, None, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9)

# Homogeneous Transformation
homogenTransMatrix = homogenTrans(R, T)

# transform the matrix and distortion coefficients to writable lists
stereo_data = { 'stereo_ret': np.asarray(stereo_ret).tolist(),
                'right_camera_matrix': np.asarray(K2).tolist(),
                'right_dist_coeff': np.asarray(D2).tolist(),
                'left_camera_matrix': np.asarray(K1).tolist(),
                'left_dist_coeff': np.asarray(D1).tolist(),
                'rotation_matrix': np.asarray(R).tolist(),
                'translation_vector': np.asarray(T).tolist(),
                'essential_matrix':np.asarray(E).tolist(),
                'fundamental_matrix':np.asarray(F).tolist(),
                'right_rotation_matrix': np.asarray(R1).tolist(),
                'left_rotation_matrix': np.asarray(R2).tolist(),
                'right_projection_coordinate_matrix': np.asarray(P1).tolist(),
                'left_projection_coordinate_matrix': np.asarray(P2).tolist(),
                'disparity_to_depth_mapping_matrix': np.asarray(Q).tolist(),
                'roi_left':np.asarray(roi_left).tolist(),
                'roi_right':np.asarray(roi_right).tolist(),
                'Homogoneous_transformation': np.asarray(homogenTransMatrix).tolist()}

# and save it to a file
with open("/home/mscv/Desktop/Internship/latestCalib/params/stereo_calibration_matrix.yaml", "w") as f:
    yaml.dump(stereo_data, f)

