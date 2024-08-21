#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
from __future__ import annotations

import rospy
from nav_msgs.msg import OccupancyGrid
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pmadmu_planner.msg import GridMap


class MapReader:
    resolution : float = 0.6  #0.6 # [m] per grid cell
    show_debug_images : bool = False
    show_debug_prints : bool = False

    def __init__(self) -> None:
        self.input_map: OccupancyGrid | None = None
        self.cvbridge : CvBridge = CvBridge()
        rospy.init_node('map_reader')
        
        rospy.Subscriber("/pmadmu_planner/merged_costmap", OccupancyGrid, self.read_map)
        #rospy.Subscriber('/map', OccupancyGrid, self.read_map)
        self.obstacles_pub : rospy.Publisher = rospy.Publisher('/pmadmu_planner/static_obstacles', OccupancyGrid, queue_size=10, latch=True)
        self.image_pub : rospy.Publisher = rospy.Publisher('/pmadmu_planner/map', Image, queue_size=10, latch=True)
        self.grid_map_pub : rospy.Publisher = rospy.Publisher('/pmadmu_planner/gridmap', GridMap, queue_size=5, latch=True)
        self.scaling_factor : float | None = None
        rospy.spin()
        return None
    

    def read_map(self, map_data : OccupancyGrid) -> None:
        if self.show_debug_prints:
            rospy.loginfo("got new data")
            rospy.loginfo(f"NEW MAP DATA ({map_data.header.frame_id}): \nsize:\n  width:\t{map_data.info.width}\n  height:\t{map_data.info.height}\nresolution:\n  {map_data.info.resolution}\n{map_data.info.origin}")
 
        self.input_map = map_data

        self.scaling_factor = map_data.info.resolution / self.resolution
        if self.show_debug_prints:
            rospy.loginfo(f"Scaling factor: {self.scaling_factor}")

        
        map_array : np.ndarray = np.array(map_data.data).reshape((map_data.info.height, map_data.info.width))
        flipped_data : np.ndarray = np.flip(map_array, axis=0)

        map_array = (255 * (1 - flipped_data / 100)).astype(np.uint8) #type:ignore


        
        #cv_image = cv2.cvtColor(map_array, cv2.COLOR_GRAY2BGR)

        #* BINARIZE
        # Apply Threshhold to enable usage of morph operators etc
        ret, orig_img_thresh = cv2.threshold(map_array, 127, 255, cv2.THRESH_BINARY)
        self.show_image(orig_img_thresh, "Orig Thresh")

        #* CLOSING 
        # Closing for denoising with a small kernel
        kernel_size : int = 1
        kernel : np.ndarray = np.ones((kernel_size, kernel_size))
        denoised_image = cv2.morphologyEx(orig_img_thresh, cv2.MORPH_CLOSE, kernel)
        self.show_image(denoised_image, "Denoised")
        
        #* ERODATION
        # we have to increase the size of the walls for them to remain after downscaling
        kernel_size : int = int(np.round(1.0/self.scaling_factor))
        kernel : np.ndarray = np.ones((kernel_size, kernel_size))
        eroded_image = cv2.erode(denoised_image, kernel)
        self.show_image(eroded_image, "Eroded")

        #* SCALING
        # Downscaling so that we get a useful gridsize for path planning. cell should be robot size
        grid_width : int = int(np.round(map_data.info.width * self.scaling_factor))
        grid_height : int = int(np.round(map_data.info.height * self.scaling_factor))
        scaled_image = cv2.resize(orig_img_thresh, (grid_width, grid_height), interpolation=cv2.INTER_LINEAR)
        self.show_image(scaled_image, "Scaled")

        #* BINARIZE
        # The scaling might result in some blurring effects, binarizing removes those
        ret, thresh_image = cv2.threshold(scaled_image, 250, 255, cv2.THRESH_BINARY)
        self.show_image(thresh_image, "Binarized")
        if self.show_debug_prints:
            rospy.loginfo(f"Resized Image to a grid size of {self.resolution}cells/m:\n  new width:\t{grid_width}\n  new height:\t{grid_height}")
        
        #* WALL RECONSTRUATION
        # Opening with different kernels to disallow walls that are only being connected by a 4er neighborhood

        #* Erosion
        kernel = np.array(
            [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]], dtype = np.uint8) #type:ignore
        eroded_result = cv2.erode(thresh_image, kernel)

        #* Dilation
        kernel = np.array(
            [[0, 1, 0],
             [1, 1, 1],
             [0, 1, 0]], dtype = np.uint8) #type:ignore
        dilated_result = cv2.dilate(eroded_result, kernel)
        
        dilated_result = thresh_image #! this skips the erosion/dilation. might want to put it back in

        self.show_image(dilated_result, "Opening")


        self.publish_image("map", dilated_result)

        flipped_map : np.ndarray = dilated_result #np.flip(dilated_result, axis=1)
        occupancy_grid_msg = OccupancyGrid()
        occupancy_grid_msg.info.width = flipped_map.shape[1]
        occupancy_grid_msg.info.height = flipped_map.shape[0]
        occupancy_grid_msg.info.resolution = self.resolution 
        occupancy_grid_msg.info.origin.position.x = 0.0 
        occupancy_grid_msg.info.origin.position.y = dilated_result.shape[0] * self.resolution
        occupancy_grid_msg.info.origin.position.z = 0.01 
        occupancy_grid_msg.info.origin.orientation.x = 1.0
        occupancy_grid_msg.info.origin.orientation.y = 0.0
        occupancy_grid_msg.info.origin.orientation.z = 0.0
        occupancy_grid_msg.info.origin.orientation.w = 0.0

        flattened_map = flipped_map.flatten()
        occupancy_grid_msg.data  = [0 if val == 0 else 100 for val in flattened_map]
        self.obstacles_pub.publish(occupancy_grid_msg)

        grid_map : GridMap = GridMap()
        grid_map.scaling_factor = self.scaling_factor
        grid_map.resolution_grid = self.resolution
        grid_map.resolution_map = map_data.info.resolution
        grid_map.grid_width = dilated_result.shape[0]
        grid_map.grid_height = dilated_result.shape[1]
        grid_map.data = dilated_result.flatten()
        self.grid_map_pub.publish(grid_map)
        return None
    

    def publish_image(self, topic_name : str, matrix : np.ndarray) -> None:
        ros_image : Image = self.cvbridge.cv2_to_imgmsg(matrix, encoding="mono8")
        self.image_pub.publish(ros_image)
        rospy.loginfo(f"updated image on topic: {topic_name}...")
        return None
    

    def show_image(self, img, name: str) -> None:
        if self.show_debug_images:
            target_size = (500, 500)  # Festlegen der Zielgröße für die Bilder
            resized_img = img #cv2.resize(img, target_size)#, interpolation=cv2.INTER_AREA)  # Skalieren des Bildes auf die Zielgröße
            cv2.imshow(name, resized_img)
            cv2.waitKey(0)
        return None
    


if __name__== '__main__':
    fb_map_reader : MapReader = MapReader()
