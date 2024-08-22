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

        self.robot_tracing : int = 50 # how many of the last robot positions should be used to clear the robots previous path from the costmap?

        self.costmaps : dict[str, OccupancyGrid] = {} # Original Costmaps. Useful since they contain infos about the data structure, header...
        self.costmap_data : dict[str, np.ndarray] = {} # Costmap Data. This is a reshaped version of the original for easier usage
        self.robot_positions : dict[str, Pose] = {}
        self.robot_pixel_positions : dict[str, list[tuple[int, int]]] = {}

        self.unique_mir_ids : list[str] = []
        self.unique_mir_ids.append(str(rospy.get_param('~robot0_name')))
        self.unique_mir_ids.append(str(rospy.get_param('~robot1_name')))
        self.unique_mir_ids.append(str(rospy.get_param('~robot2_name')))
        self.unique_mir_ids.append(str(rospy.get_param('~robot3_name')))
        rospy.loginfo(f"[CController] Registered {len(self.unique_mir_ids)} mir bots, with the IDs: {self.unique_mir_ids}")


        self.costmap_subscribers : list[rospy.Subscriber] = [rospy.Subscriber(f"/{robot_name}/move_base_flex/global_costmap/costmap", OccupancyGrid, self.read_costmap, callback_args=robot_name) for robot_name in self.unique_mir_ids]
        self.costmap_update_subscribers : list[rospy.Subscriber] = [rospy.Subscriber(f"/{robot_name}/move_base_flex/global_costmap/costmap_updates", OccupancyGridUpdate, self.update_costmap, callback_args=robot_name) for robot_name in self.unique_mir_ids]
        self.robot_position_subscribers_real : list[rospy.Subscriber] = [rospy.Subscriber(f'/{robot_name}/robot_pose', Pose, self.update_robot_pose, callback_args=robot_name) for robot_name in self.unique_mir_ids]
        self.robot_position_subscribers : list[rospy.Subscriber] = [rospy.Subscriber(f'/{robot_name}/mir_pose_simple', Pose, self.update_robot_pose, callback_args=robot_name) for robot_name in self.unique_mir_ids]

        self.clear_services : list[rospy.ServiceProxy] = [rospy.ServiceProxy(f"/{robot_name}/move_base_flex/clear_costmaps", Empty) for robot_name in self.unique_mir_ids]

        rospy.Timer(rospy.Duration.from_sec(0.5), self.clear_costmaps, oneshot=False)
        rospy.Timer(rospy.Duration.from_sec(0.5), self.merge_costmaps, oneshot=False)
        rospy.Timer(rospy.Duration.from_sec(0.1), self.log_robot_positons, oneshot=False)
        self.merged_costmap_publisher : rospy.Publisher = rospy.Publisher("/pmadmu_planner/merged_costmap", OccupancyGrid, queue_size=5, latch=True)
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
    

    def remove_robots_from_costmap_via_blobs(self, costmap: np.ndarray, ignore_robot : str) -> np.ndarray:
        #! deprecated; not a good idea to use this function since the blobs are unreliable and may not cover the entire robot  
        # remove invalid pixels
        valid_pixels = costmap >= 0
        costmap_masked = np.where(valid_pixels, costmap, 0)

        image : np.ndarray = np.uint8(costmap_masked) # type: ignore

        _, binary = cv2.threshold(image, 5, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            for robot_name, position in self.robot_pixel_positions.items():
                if robot_name == ignore_robot:
                    continue
                is_inside_contour : bool = cv2.pointPolygonTest(contour, position, measureDist=False) >= 0 # type: ignore since stubs are missing
                if is_inside_contour:
                    cv2.drawContours(image, [contour], -1, 0, cv2.FILLED) # type: ignore since stubs are missing
                    continue
        return np.int8(image) # type: ignore
    

    def remove_robots_from_costmap(self, costmap: np.ndarray, ignore_robot : str) -> np.ndarray:
        if ignore_robot not in self.costmaps:
            return costmap
        valid_pixels = costmap >= 0
        costmap_masked = np.where(valid_pixels, costmap, 0)
        image : np.ndarray = np.uint8(costmap_masked) # type: ignore
        
        robot_size : float = 2.5
        clearing_radius : int = int(np.round(robot_size / self.costmaps[ignore_robot].info.resolution)) # todo: make this a config parameter
        for robot_name, positions in self.robot_pixel_positions.items():
            if robot_name == ignore_robot:
                continue
            for position in positions:
                cv2.circle(image, position, clearing_radius, color=0, thickness=-1)  # type: ignore
        return np.int8(image) # type: ignore
    

    
    def update_robot_pose(self, pose: Pose, robot_name : str = "") -> None:
        if robot_name == "":
            rospy.logwarn("[CMap Reader]: Invalid Robot Name in Pose Update")
            return None
        if robot_name not in self.costmaps:
            return None
        self.robot_positions[robot_name] = pose
        return None
    

    def log_robot_positons(self, _) -> None:
        for robot_name, position in self.robot_positions.items():
            if robot_name not in self.robot_pixel_positions.keys():
                self.robot_pixel_positions[robot_name] = []
            # Convert Pose to Pixel Position
            x_pos : int = int(np.round(position.position.x / self.costmaps[robot_name].info.resolution))
            y_pos : int = int(np.round(position.position.y / self.costmaps[robot_name].info.resolution))
            self.robot_pixel_positions[robot_name].append((x_pos, y_pos))
            if len(self.robot_pixel_positions[robot_name]) > self.robot_tracing:
                self.robot_pixel_positions[robot_name].pop(0)
        return None


    def update_costmap(self, costmap_update: OccupancyGridUpdate, robot_name : str = "") -> None:
        # Costmap Update. Does trigger regularly but only contains data from robots proximity. have to merge it with the initial costmap
        if robot_name not in self.costmaps or robot_name not in self.costmap_data:
            rospy.logwarn(f"[CMap Reader]: Robot Name {robot_name} is invalid; May not be fully initialized yet")
            return None
        self.costmaps[robot_name].data = costmap_update.data

        costmap = self.costmap_data[robot_name].copy()
        costmap[costmap_update.y:costmap_update.y+costmap_update.height, costmap_update.x:costmap_update.x+costmap_update.width] = np.reshape(costmap_update.data, (costmap_update.height, costmap_update.width))
        self.costmap_data[robot_name] = self.remove_robots_from_costmap(costmap, robot_name)
        return None
    

    def read_costmap(self, costmap : OccupancyGrid, robot_name : str = "") -> None:
        # Initial Costmap Init. Does only trigger once so we have to update the costmap manually using update_costmap()
        if robot_name == "":
            rospy.logwarn(f"[CMap Reader]: Robot Name is invalid")
            return None
        rospy.loginfo(f"[CMap Reader]: Received a Map for Robot {robot_name}")
        self.costmaps[robot_name] = costmap
        costmap_data : np.ndarray = np.reshape(costmap.data, (costmap.info.height, costmap.info.width))
        self.costmap_data[robot_name] = costmap_data
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