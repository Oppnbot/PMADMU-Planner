#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
from __future__ import annotations

import rospy
from nav_msgs.msg import OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
from geometry_msgs.msg import Pose
import numpy as np
import re
import cv2
from std_srvs.srv import Empty


class CostMapReader:
    def __init__(self) -> None:
        rospy.init_node('CostMapReader')

        self.robot_tracing : int = 0 # how many of the last robot positions should be used to clear the robots previous path from the costmap?

        self.costmaps : dict[int, OccupancyGrid] = {} # Original Costmaps. Useful since they contain infos about the data structure, header...
        self.costmap_data : dict[int, np.ndarray] = {} # Costmap Data. This is a reshaped version of the original for easier usage
        self.robot_positions : dict[int, Pose] = {}
        self.robot_pixel_positions : dict[int, list[tuple[int, int]]] = {}

        # Scan Topics for Mir-Robots to create subscribers. ALL ROBOTS MUST BE NAMED LIKE THIS: mir1
        topics = rospy.get_published_topics()
        self.unique_mir_ids : set[int] = set()
        pattern = r"/mir(\d+)/"
        for topic in topics:
            match_string = re.search(pattern, str(topic))
            if match_string:
                robot_id = int(match_string.group(1))
                self.unique_mir_ids.add(robot_id)

        self.costmap_subscribers : list[rospy.Subscriber] = [rospy.Subscriber(f"/mir{robot_id}/move_base_flex/global_costmap/costmap", OccupancyGrid, self.read_costmap, callback_args=robot_id) for robot_id in self.unique_mir_ids]
        self.costmap_update_subscribers : list[rospy.Subscriber] = [rospy.Subscriber(f"/mir{robot_id}/move_base_flex/global_costmap/costmap_updates", OccupancyGridUpdate, self.update_costmap, callback_args=robot_id) for robot_id in self.unique_mir_ids]
        self.robot_position_subscribers_real : list[rospy.Subscriber] = [rospy.Subscriber(f'/mir{robot_id}/robot_pose', Pose, self.update_robot_pose, callback_args=robot_id) for robot_id in self.unique_mir_ids]
        self.robot_position_subscribers : list[rospy.Subscriber] = [rospy.Subscriber(f'/mir{robot_id}/mir_pose_simple', Pose, self.update_robot_pose, callback_args=robot_id) for robot_id in self.unique_mir_ids]

        self.clear_services : list[rospy.ServiceProxy] = [rospy.ServiceProxy(f"/mir{robot_id}/move_base_flex/clear_costmaps", Empty) for robot_id in self.unique_mir_ids]

        #rospy.Timer(rospy.Duration.from_sec(0.5), self.clear_costmaps, oneshot=False)
        rospy.Timer(rospy.Duration.from_sec(0.5), self.merge_costmaps, oneshot=False)
        rospy.Timer(rospy.Duration.from_sec(0.1), self.log_robot_positons, oneshot=False)
        self.merged_costmap_publisher : rospy.Publisher = rospy.Publisher("/formation_builder/merged_costmap", OccupancyGrid, queue_size=5, latch=True)
        return None


    def merge_costmaps(self, _) -> OccupancyGrid | None:
        if len(self.costmaps.values()) < 1:
            rospy.loginfo("[CMap Reader]: No Costmaps Available")
            return None
        # convert vector to a numpy array for easier merging of different sized maps
        height : int = 0
        width : int = 0
        for costmap in self.costmaps.values():
            if costmap.info.width > width:
                width = costmap.info.width
            if costmap.info.height > height:
                height = costmap.info.height
        
        merged_data : np.ndarray = np.zeros((height, width))
        for costmap_data in self.costmap_data.values():
            merged_data = np.maximum(merged_data, costmap_data)

        merged_costmap : OccupancyGrid = OccupancyGrid()
        merged_costmap.info = list(self.costmaps.values())[-1].info
        merged_costmap.info.height = height
        merged_costmap.info.width = width

        merged_costmap.data = merged_data.flatten().astype(int).tolist()
        rospy.loginfo("[CMap Reader]: Publishing Merged Costmap...")
        self.merged_costmap_publisher.publish(merged_costmap)
        return None
    

    def remove_robots_from_costmap_via_blobs(self, costmap: np.ndarray, ignore_robot : int) -> np.ndarray:
        #! deprecated; not a good idea to use this function since the blobs are unreliable and may not cover the entire robot  
        # remove invalid pixels
        valid_pixels = costmap >= 0
        costmap_masked = np.where(valid_pixels, costmap, 0)

        image : np.ndarray = np.uint8(costmap_masked) # type: ignore

        _, binary = cv2.threshold(image, 5, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            for robot_id, position in self.robot_pixel_positions.items():
                if robot_id == ignore_robot:
                    continue
                is_inside_contour : bool = cv2.pointPolygonTest(contour, position, measureDist=False) >= 0 # type: ignore since stubs are missing
                if is_inside_contour:
                    cv2.drawContours(image, [contour], -1, 0, cv2.FILLED) # type: ignore since stubs are missing
                    continue
        return np.int8(image) # type: ignore
    

    def remove_robots_from_costmap(self, costmap: np.ndarray, ignore_robot : int) -> np.ndarray:
        if ignore_robot not in self.costmaps:
            return costmap
        valid_pixels = costmap >= 0
        costmap_masked = np.where(valid_pixels, costmap, 0)
        image : np.ndarray = np.uint8(costmap_masked) # type: ignore
        
        robot_size : float = 2.5
        clearing_radius : int = int(np.round(robot_size / self.costmaps[ignore_robot].info.resolution)) # todo: make this a config parameter
        for robot_id, positions in self.robot_pixel_positions.items():
            if robot_id == ignore_robot:
                continue
            for position in positions:
                cv2.circle(image, position, clearing_radius, color=0, thickness=-1)  # type: ignore
        return np.int8(image) # type: ignore
    

    
    def update_robot_pose(self, pose: Pose, robot_id : int = -1) -> None:
        if robot_id == -1:
            rospy.logwarn("[CMap Reader]: Invalid Robot ID in Pose Update")
            return None
        if robot_id not in self.costmaps:
            return None
        self.robot_positions[robot_id] = pose
        return None
    

    def log_robot_positons(self, _) -> None:
        for robot_id, position in self.robot_positions.items():
            if robot_id not in self.robot_pixel_positions.keys():
                self.robot_pixel_positions[robot_id] = []
            # Convert Pose to Pixel Position
            x_pos : int = int(np.round(position.position.x / self.costmaps[robot_id].info.resolution))
            y_pos : int = int(np.round(position.position.y / self.costmaps[robot_id].info.resolution))
            self.robot_pixel_positions[robot_id].append((x_pos, y_pos))
            if len(self.robot_pixel_positions[robot_id]) > self.robot_tracing:
                self.robot_pixel_positions[robot_id].pop(0)
        return None


    def update_costmap(self, costmap_update: OccupancyGridUpdate, robot_id : int = -1) -> None:
        # Costmap Update. Does trigger regularly but only contains data from robots proximity. have to merge it with the initial costmap
        if robot_id not in self.costmaps or robot_id not in self.costmap_data:
            rospy.logwarn(f"[CMap Reader]: Robot ID {robot_id} is invalid; May not be fully initialized yet")
            return None
        self.costmaps[robot_id].data = costmap_update.data

        costmap = self.costmap_data[robot_id].copy()
        costmap[costmap_update.y:costmap_update.y+costmap_update.height, costmap_update.x:costmap_update.x+costmap_update.width] = np.reshape(costmap_update.data, (costmap_update.height, costmap_update.width))
        self.costmap_data[robot_id] = self.remove_robots_from_costmap(costmap, robot_id)
        return None
    

    def read_costmap(self, costmap : OccupancyGrid, robot_id : int = -1) -> None:
        # Initial Costmap Init. Does only trigger once so we have to update the costmap manually using update_costmap()
        if robot_id == -1:
            rospy.logwarn(f"[CMap Reader]: Robot ID {robot_id} is invalid")
            return None
        rospy.loginfo(f"[CMap Reader]: Received a Map for Robot {robot_id}")
        self.costmaps[robot_id] = costmap
        costmap_data : np.ndarray = np.reshape(costmap.data, (costmap.info.height, costmap.info.width))
        self.costmap_data[robot_id] = costmap_data
        #todo: remove robots and robot trails from costmap
        return None


    def clear_costmaps(self, _) -> None:
        rospy.loginfo("[CMap Reader]: Clearing Costmaps...")
        for clear_service in self.clear_services:
            clear_service()
        return None


if __name__== '__main__':
    fb_map_reader : CostMapReader = CostMapReader()
    rospy.spin()
