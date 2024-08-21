#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#! this requires grid map package from https://github.com/ethz-asl/grid_map

from pyclbr import Class
import rospy
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import MapMetaData
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import random
from visualization_msgs.msg import Marker


def map_callback(data):

    rospy.loginfo("got new data")

    rospy.loginfo(f"map size w: {data.info.width}, h: {data.info.height}")

    map_data = np.array(data.data).reshape((data.info.height, data.info.width))
    my_map : np.ndarray = np.zeros((data.info.width, data.info.height),dtype=np.uint8)

    for y in range(data.info.height-1):
        for x in range(data.info.width-1):
            my_map[x, y] = abs(map_data[y, x]) * 255

    my_map = cv2.bitwise_not(my_map)

    new_width = int(my_map.shape[1] / 2)
    new_height = int(my_map.shape[0] / 2)
    resized_image = cv2.resize(my_map, (new_width, new_height))

    bridge = CvBridge()
    image_msg = bridge.cv2_to_imgmsg(resized_image, encoding="mono8")

    image_pub.publish(image_msg)





    rospy.loginfo("Done")
    #cv2.imshow("map", resized_image)
    #cv2.waitKey(0)
    return None






if __name__ == '__main__':
    image_pub = rospy.Publisher('/my_grid_img', Image, queue_size=10)
    grid_pub = rospy.Publisher('/my_grid_grid', Marker, queue_size=10)
    rospy.init_node('map_display_node')
    rospy.Subscriber('/map', OccupancyGrid, map_callback)
    rospy.spin()
