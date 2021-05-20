#!/usr/bin/env python3

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
import ros_numpy
import sensor_msgs.point_cloud2 as pc2


N = 2
modelname = "3dObjectMic"

"""
    Set the detector for the surface matching and training the detector
"""
detector = cv2.ppf_match_3d_PPF3DDetector(0.025, 0.05)

rospy.loginfo('Loading model...')
pc = cv2.ppf_match_3d.loadPLYSimple("/home/mscv/Desktop/Internship/3D_model/surface_matching/%s.ply" %modelname, 1)
#rospy.loginfo(pc.shape)

rospy.loginfo('Training...')
#detector.trainModel(pc)

rospy.loginfo("Training has completed")


# Declare the publisher for stereo point cloud
depthPoint_pub = rospy. Publisher('/3dmatching/matchObj', PointCloud2, queue_size=1)


DIST_COLORS = [\
    "#2980b9",\
    "#27ae60",\
    "#f39c12",\
    "#ff0000",\
    ]

DIST_COLOR_LEVELS = 20


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

def convert_numpy_2_pointcloud2_color(points, stamp=None, frame_id=None, maxDistColor=None):
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
        header.frame_id = "/map"
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

def computeMatch(scene):

    # pick a height
    height =  int (scene.height / 2)
    # pick x coords near front and center
    middle_x = int (scene.width / 2)
    # examine point
    middle, scenePCL = read_depth (middle_x, height, scene)
    # do stuff with middle
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(scenePCL)
    scenePCL = cv2.ppf_match_3d.loadPLYSimple(pcd, 1)
    rospy.loginfo(scenePCL.shape)
    #o3d.visualization.draw_geometries([pcd])

    #rospy.loginfo(scenePCL.shape)
    #scenePCL = scene
    #xyz_array = ros_numpy.point_cloud2.get_xyz_points(scenePCL)
    #matchPub(xyz_array)



    # Find the matching for the scene we have with our detector
    rospy.loginfo('Matching...')

    resultsDetect = detector.match(pcd.points, 1.0/40.0, 0.05)
    
    print('Performing ICP...')
    icp = cv2.ppf_match_3d_ICP(100)
    _, results = icp.registerModelToScene(pc, scenePCL, resultsDetect[:N])

    print("Poses: ")
    for i, result in enumerate(results):
        result.printPose()
        print("\n-- Pose to Model Index %d: NumVotes = %d, Residual = %f\n%s\n" % (result.modelIndex, result.numVotes, result.residual, result.pose))
        if i == 0:
            pct = cv2.ppf_match_3d.transformPCPose(pc, result.pose)
    
    rospy.loginfo('Finish reprojecting the Matching 3d object')
    point_cloud2 = convert_numpy_2_pointcloud2_color(pct)
    matchPub.publish(point_cloud2)
    
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

def read_depth(width, height, data) :
    # read function
    if (height >= data.height) or (width >= data.width) :
        return -1
    data_out = pc2.read_points(data, field_names=None, skip_nans=False, uvs=[[width, height]])
    int_data = next(data_out)
    rospy.loginfo("int_data " + str(int_data))
    test_data = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(data)
    return int_data, test_data

def matchPub(depthPoint):
    depthPoint_pub.publish(depthPoint)



def listener():
    rospy.init_node('match_obj', anonymous=True)
    #scene = message_filters.Subscriber('/stereo/depthPoint', PointCloud2)
    #mic2 = message_filters.Subscriber('/camera_00/image_raw', Image)
    #ts = message_filters.ApproximateTimeSynchronizer([scene], 10, 5)
    #ts.registerCallback(computeMatch)
    scene = rospy.Subscriber('/stereo/depthPoint', PointCloud2, computeMatch)
    #computeMatch(scene)
    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass
