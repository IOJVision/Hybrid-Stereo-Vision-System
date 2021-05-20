#!/usr/bin/env python

import rospy
import numpy as np
import cv2
import message_filters
from cv_bridge import CvBridge
from std_msgs.msg import UInt8, Float64, Header
import std_msgs.msg
from sensor_msgs.msg import Image, CompressedImage, PointCloud2, PointField
import yaml
import open3d as o3d
from open3d import *
from numpy import random as rnd
from sensor_msgs import point_cloud2
import sensor_msgs.point_cloud2 as pcl2
import struct



# Declare the publisher for diparity map 
disp_img_pub = rospy.Publisher('/stereo/disparity', Image, queue_size=1)

# Declare the publisher for mic 1 undistort
mic1_undist_pub = rospy.Publisher('/stereo/undistort/mic1', Image, queue_size=1)

# Declare the publisher for mic 2 undistort
mic2_undist_pub = rospy.Publisher('/stereo/undistort/mic2', Image, queue_size=1)

# Declare the publisher for stereo point cloud
depthPoint_pub = rospy. Publisher('/stereo/depthPoint', PointCloud2, queue_size=1)

# Declare the publisher for stereo point cloud color
depthPointColor_pub = rospy. Publisher('/stereo/depthPointColor', Image, queue_size=1)

out_points = []
out_colors = []

DIST_COLORS = [\
    "#2980b9",\
    "#27ae60",\
    "#f39c12",\
    "#ff0000",\
    ]

DIST_COLOR_LEVELS = 20

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def choose(self):

    choice='q'

    rospy.loginfo("|-------------------------------|")
    rospy.loginfo("|'0': Save  ")
    rospy.loginfo("|'q': Quit ")
    rospy.loginfo("|-------------------------------|")
    choice = input()
    return choice

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

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

def calc_2nd_depth_map(imgL, imgR):
    # disparity range is tuned for 'aloe' image pair
    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 1,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 150,
        speckleRange = 32
    )

    print('computing disparity...')
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    return disp

def hex_to_RGB(hex):
    ''' "#FFFFFF" -> [255,255,255] '''
    # Pass 16 to the integer function for change of base
    return [int(hex[i:i+2], 16) for i in range(1,6,2)]


def RGB_to_hex(RGB):
    ''' [255,255,255] -> "#FFFFFF" '''
    # Components need to be integers for hex to make sense
    RGB = [int(x) for x in RGB]
    return "#"+"".join(["0{0:x}".format(v) if v < 16 else
            "{0:x}".format(v) for v in RGB])

def color_dict(gradient):
    ''' Takes in a list of RGB sub-lists and returns dictionary of
    colors in RGB and hex form for use in a graphing function
    defined later on '''
    return {"hex":[RGB_to_hex(RGB) for RGB in gradient],
        "r":[RGB[0] for RGB in gradient],
        "g":[RGB[1] for RGB in gradient],
        "b":[RGB[2] for RGB in gradient]}

def linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
    ''' returns a gradient list of (n) colors between
    two hex colors. start_hex and finish_hex
    should be the full six-digit color string,
    inlcuding the number sign ("#FFFFFF") '''
    # Starting and ending colors in RGB form
    s = hex_to_RGB(start_hex)
    f = hex_to_RGB(finish_hex)
    # Initilize a list of the output colors with the starting color
    RGB_list = [s]
    # Calcuate a color at each evenly spaced value of t from 1 to n
    for t in range(1, n):
        # Interpolate RGB vector for color at the current value of t
        curr_vector = [
            int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
            for j in range(3)
        ]
        # Add it to our list of output colors
        RGB_list.append(curr_vector)

    return color_dict(RGB_list)

def rand_hex_color(num=1):
    ''' Generate random hex colors, default is one,
        returning a string. If num is greater than
        1, an array of strings is returned. '''
    colors = [
        RGB_to_hex([x*255 for x in rnd.rand(3)])
        for i in range(num)
    ]
    if num == 1:
        return colors[0]
    else:
        return colors


def polylinear_gradient(colors, n):
    ''' returns a list of colors forming linear gradients between
        all sequential pairs of colors. "n" specifies the total
        number of desired output colors '''
    # The number of colors per individual linear gradient
    n_out = int(float(n) / (len(colors) - 1))
    # returns dictionary defined by color_dict()
    gradient_dict = linear_gradient(colors[0], colors[1], n_out)

    if len(colors) > 1:
        for col in range(1, len(colors) - 1):
            next = linear_gradient(colors[col], colors[col+1], n_out)
            for k in ("hex", "r", "g", "b"):
                # Exclude first point to avoid duplicates
                gradient_dict[k] += next[k][1:]

    return gradient_dict



def color_map(data, colors, nLevels):
    # Get the color gradient dict.
    gradientDict = polylinear_gradient(colors, nLevels)

    # Get the actual levels generated.
    n = len( gradientDict["hex"] )

    # Level step.
    dMin = data.min()
    dMax = data.max()
    step = ( dMax - dMin ) / (n-1)

    stepIdx = ( data - dMin ) / step
    stepIdx = stepIdx.astype(np.int)

    rArray = np.array( gradientDict["r"] )
    gArray = np.array( gradientDict["g"] )
    bArray = np.array( gradientDict["b"] )

    r = rArray[ stepIdx ]
    g = gArray[ stepIdx ]
    b = bArray[ stepIdx ]

    return r, g, b

def convert_numpy_2_pointcloud2_color(points, out_colors, stamp=None, frame_id=None, maxDistColor=None):
    # Clipping input.
    dist = np.linalg.norm( points, axis=1 )
    if ( maxDistColor is not None and maxDistColor > 0):
        dist = np.clip(dist, 0, maxDistColor)

    # Compose color.
    cr, cg, cb = color_map( dist, DIST_COLORS, DIST_COLOR_LEVELS )

    C = np.zeros((cr.size, 4), dtype=np.uint8) + 255

    C[:, 0] = cb.astype(np.uint8)
    C[:, 1] = cg.astype(np.uint8)
    C[:, 2] = cr.astype(np.uint8)

    C = C.view("uint32")

    # 2ND Try compose color
    #D = np.zeros((cr.size, 4), dtype=np.uint8) + 255

    #D[:, 0] = out_colors[:, 0].reshape((-1, 1))
    #D[:, 1] = out_colors[:, 1].reshape((-1, 1))
    #D[:, 2] = out_colors[:, 2].reshape((-1, 1))

    #D = D.view("uint32")


    # Structured array.
    pointsColor = np.zeros( (points.shape[0], 1), \
        dtype={ 
            "names": ( "x", "y", "z", "rgba" ), 
            "formats": ( "f4", "f4", "f4", "u4" )} )

    points = points.astype(np.float32)

    pointsColor["x"] = points[:, 0].reshape((-1, 1))
    pointsColor["y"] = points[:, 1].reshape((-1, 1))
    pointsColor["z"] = points[:, 2].reshape((-1, 1))
    pointsColor["rgba"] = C

    header = Header()

    if stamp is None:
        header.stamp = rospy.Time().now()
    else:
        header.stamp = stamp

    if frame_id is None:
        header.frame_id = "/my_frame"
    else:
        header.frame_id = frame_id

    msg = PointCloud2()
    msg.header = header

    if len(points.shape) == 3:
        msg.height = points.shape[1]
        msg.width = points.shape[0]
    else:
        msg.height = 1
        msg.width  = points.shape[0]

    msg.fields = [
        PointField('x',  0, PointField.FLOAT32, 1),
        PointField('y',  4, PointField.FLOAT32, 1),
        PointField('z',  8, PointField.FLOAT32, 1),
        PointField('rgb', 12, PointField.UINT32, 1),
        ]

    msg.is_bigendian = False
    msg.point_step   = 16
    msg.row_step     = msg.point_step * points.shape[0]
    msg.is_dense     = int( np.isfinite(points).all() )
    msg.data         = pointsColor.tostring()

    

    return msg


def depth_map(mic1, mic2):
    bridge = CvBridge()

    mic1_img = bridge.imgmsg_to_cv2(mic1, "bgr8")
    mic2_img = bridge.imgmsg_to_cv2(mic2, "bgr8")

    # Load the Stereo Calibration Data
    loadStereoCalib = yaml.load(open('/home/mscv/Desktop/Internship/0804_circleCalib/data/stereo3D/params/3D_stereo_calibration_matrix.yaml'))

    # Load The K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q
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
    #newdisp = calc_2nd_depth_map(img_1_undistorted, img_2_undistorted)
    #projImg(newdisp)

    #Publish 3D Point Cloud
    points = cv2.reprojectImageTo3D(newdisp, Q)
    colors = cv2.cvtColor(mic1_img, cv2.COLOR_BGR2RGB)
    mask = newdisp > newdisp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    
    point_cloud = convert_numpy_2_pointcloud2_color(out_points, out_colors)

    depthPoint_pub.publish(point_cloud)

    # Testing new cloud
    """
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'map'
    #create pcl from points
    scaled_polygon_pcl = pcl2.create_cloud_xyz32(header, out_points)
    #publish    
    rospy.loginfo("happily publishing sample pointcloud.. !")
    depthPoint_pub.publish(scaled_polygon_pcl)

    """
    #Publish undistort image
    # We need grayscale for disparity map.
    gray_left = cv2.cvtColor(img_1_undistorted, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_2_undistorted, cv2.COLOR_BGR2GRAY)
    mic1Und(gray_left)
    mic2Und(gray_right)
    
    if KeyboardInterrupt:
        savePly(out_points, out_colors)

    

def savePly(out_points, out_colors):
    #out_fn = '/home/mscv/Desktop/out.ply'
    #write_ply(out_fn, out_points, out_colors)
    #rospy.loginfo('%s saved' % out_fn)
    rospy.loginfo("Shutting Down")

def projImg(dispImg):
    bridge = CvBridge()
    disp_img_pub.publish(bridge.cv2_to_imgmsg(dispImg))


def mic1Und(mic1Img):
    bridge = CvBridge()
    mic1_undist_pub.publish(bridge.cv2_to_imgmsg(mic1Img))

def mic2Und(mic2Img):
    bridge = CvBridge()
    mic2_undist_pub.publish(bridge.cv2_to_imgmsg(mic2Img))

def depthPub(depthPoint):
    bridge = CvBridge()
    depthPoint_pub.publish(depthPoint)

def colorDepthPub(depthColor):
    bridge = CvBridge()
    depthPointColor_pub.publish(bridge.cv2_to_imgmsg(depthColor))


def listener():
    rospy.init_node('disp_stereo', anonymous=True)
    mic1 = message_filters.Subscriber('/camera_02/image_raw', Image)
    mic2 = message_filters.Subscriber('/camera_00/image_raw', Image)
    ts = message_filters.ApproximateTimeSynchronizer([mic1, mic2], 10, 5)
    ts.registerCallback(depth_map)
    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass
