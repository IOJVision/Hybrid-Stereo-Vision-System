import numpy as np
import cv2
import glob
import yaml

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#row, original 7 but put 6, a
#column, original 9 but put 8, b
a = 6
b = 8


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((a*b,3), np.float32)
objp[:,:2] = np.mgrid[0:b,0:a].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

#FOR Perspective Camera
images = glob.glob('/home/mscv/Desktop/Internship/latestCalib/data/cam/*.jpg')


for fname in images:
    img = cv2.imread(fname)
    #cv2.imshow('img',img)
    #cv2.waitKey(0000)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (b,a),None)
    print(ret)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (b,a), corners2,ret)
        #cv2.imshow('img',img)
        #cv2.waitKey(5000)
    else:
        print("Not found")
    
    print(ret)

cv2.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

print("ret: ", ret)
print("mtx: \n", mtx)# Parameter matrix
print("dist: \n", dist)# Distortion coefficient distortion coefficients = (k_1,k_2,p_1,p_2,k_3)
print("rvecs: \n", rvecs)# Vector or External parameters
print("tvecs: \n", tvecs)# Translation vector or Outer parameter

# solvePnP to get the rvec matrix and tvec matrix
retval, rvec_mat, tvec_mat = cv2.solvePnP(objp, corners2_left, mtx_left, dist_left)

#Re-projection error gives a good estimation of just how exact the found parameters are
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )
total_error = mean_error/len(objpoints)
print(total_error)

# transform the matrix and distortion coefficients to writable lists
data = {'camera_matrix': np.asarray(mtx).tolist(),
        'dist_coeff': np.asarray(dist).tolist(),
        'total_error': np.asarray(total_error).tolist(),
        'object_point': np.asarray(objpoints).tolist(),
        'image_point': np.asarray(imgpoints).tolist(),
        'rvecs': np.asarray(rvecs).tolist(),
        'tvecs': np.asarray(tvecs).tolist(),
        'objp': np.asarray(objp).tolist(),
        'corners2': np.asarray(corners2).tolist(),
        'rvec_mat': np.asarray(rvec_mat).tolist(),
        'tvec_mat': np.asarray(tvec_mat).tolist()}

# and save it to a file
with open("/home/mscv/Desktop/Internship/latestCalib/params/cam_calibration_matrix.yaml", "w") as f:
    yaml.dump(data, f)
