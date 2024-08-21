#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from operator import index

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import time
import heapq

class WavefrontExpansionNode:
    def __init__(self):
        rospy.init_node('wavefront_expansion_node')
        rospy.Subscriber('/formation_builder/map', Image, self.image_callback)
        self.timing_pub: rospy.Publisher = rospy.Publisher("/timing_image", Image, queue_size=1, latch=True)
        self.image_pub = rospy.Publisher("/path_image", Image, queue_size=1, latch=True)
        self.path_pub = rospy.Publisher("/path_output", Path, queue_size=10, latch=True)
        self.grid_resolution = 2.0
        self.cv_bridge = CvBridge()
        self.point_size : int = 2

        self.live_vizualisation : bool = True
        self.allow_diagonals : bool = False

        self.occupied_from : np.ndarray | None = None
        self.occupied_until: np.ndarray | None = None



    def image_callback(self, img_msg):
        img = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding="mono8")
        grid = self.process_image(img)

        # for 0.5m res
        #start_position : tuple[int, int] = (45, 25)
        #goal_position : tuple[int, int] = (120, 120)

        # for
        #start_position : tuple[int, int] = (25, 12)
        #goal_position : tuple[int, int] = (47, 17)

        start_position : tuple[int, int] =(20, 16)
        goal_position : tuple[int, int] = (48, 50)

        path = self.wavefront_expansion(grid, start_position, goal_position)

        #rospy.loginfo(f"path {path}")
        #path_img = self.draw_path_on_image_colored(img, path, start_position, goal_position)
        #path_image_msg = self.cv_bridge.cv2_to_imgmsg(path_img, encoding="bgr8")
        #self.image_pub.publish(path_image_msg)
        return None

    def process_image(self, img):
        # Umwandlung des Bildes in ein Rastergitter
        grid : np.ndarray = np.array(img) // 255
        return grid.tolist()
    

    def wavefront_expansion(self, static_obstacles:np.ndarray, start_pos:tuple[int, int], goal_pos:tuple[int, int]):
        rospy.loginfo("Starting wavefront expansion")
        start_time = time.time()

        

        heap : list[tuple[float, tuple[int, int]]] = [(0, start_pos)]
        rows = len(static_obstacles)
        cols = len(static_obstacles[0])

        visited = set()
        direct_neighbors : list[tuple[int, int]] = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        diagonal_neighbors: list[tuple[int, int]] = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
        neighbors: list[tuple[int, int]] = direct_neighbors
        if self.allow_diagonals:
            neighbors += diagonal_neighbors

        timings : np.ndarray = np.zeros((rows, cols))-1
        
        queue = [start_pos]


        #timings[30,30] = 200
        #queue = [start_pos, (30, 30)]
        
        
        iterations : int = 0

        timings[start_pos[0], start_pos[1]] = 0.01 #? is this necessary?

        current_element : tuple[int, int]
        while heap:
            if not queue:
                rospy.loginfo("Queue ist empty!")
                break

            iterations += 1
            current_element = queue.pop(0)

            
            if iterations % 100 == 0:
                rospy.loginfo(f"{iterations} done!")

            if self.live_vizualisation:
                self.draw_timings(timings, static_obstacles, start_pos, goal_pos)

            if iterations > 500000:
                rospy.loginfo("breaking because algorithm reached max iterations")
                break

            if current_element == goal_pos:
                rospy.loginfo(f"Reached goal but queue is not empty")
                continue
                rospy.loginfo(f"Reached the goal after {iterations} iterations")
                break

            #cost_increase : float = 1
            current_cost = timings[current_element[0], current_element[1]]

            for x_neighbor, y_neighbor in neighbors:
                x, y = current_element[0] + x_neighbor, current_element[1] + y_neighbor
                if 0 <= x < rows and 0 <= y < cols and static_obstacles[x][y] != 0:# and (x, y) not in visited:
                    driving_cost : float = current_cost + (1 if abs(x_neighbor+y_neighbor) == 1 else 1.41421366)
                    #rospy.loginfo(f"driving cost{driving_cost}, current_cost {timings[x,y]} ")
                    if driving_cost < timings[x, y] or timings[x,y] < 0:
                        timings[x,y] = driving_cost
                        queue.append((x, y))
        rospy.loginfo(f"stopped after a total of {iterations} iterations")

                    

        
        end_time = time.time()
        elapsed_time = end_time - start_time
        rospy.loginfo(f"It took {elapsed_time} s.")

        #print_list = timings.tolist()
        #rospy.loginfo(print_list)
        
        #rospy.loginfo(f"final map\n{timings}")

        path: list[tuple[int, int]] = self.find_path(timings, static_obstacles, start_pos, goal_pos)
        rospy.loginfo(f"shortest path consists of {len(path)} nodes")
        self.draw_timings(timings, static_obstacles, start_pos, goal_pos, path)
        return None
    
    def find_path(self, timings: np.ndarray, static_obstacles:np.ndarray, start_pos:tuple[int, int], goal_pos:tuple[int, int]) -> list[tuple[int, int]]:
        rospy.loginfo("searching for the shortest path...")
        path : list[tuple[int, int]] = []
        next_pos : tuple[int, int] = goal_pos

        direct_neighbors : list[tuple[int, int]] = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        diagonal_neighbors: list[tuple[int, int]] = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
        neighbors: list[tuple[int, int]] = direct_neighbors
        if self.allow_diagonals:
            neighbors += diagonal_neighbors

        rows = len(timings)
        cols = len(timings[0])


        while next_pos != start_pos:
            if self.live_vizualisation:
                self.draw_timings(timings, static_obstacles, start_pos, goal_pos, path)
            lowest_timing : float = float('inf')
            lowest_neigbor: tuple[int, int]

            for x_neighbor, y_neighbor in neighbors:
                x, y = next_pos[0] + x_neighbor, next_pos[1] + y_neighbor
                if 0 <= x < rows and 0 <= y < cols and -1 < timings[x, y] < lowest_timing:
                    lowest_timing = timings[x, y]
                    lowest_neigbor = (x, y)
            if lowest_timing == float('inf'):
                # This error means that goal may be unreachable. should not occur if wavefront generation succeeded
                rospy.logerr("pathfinding failed! there is no valid neighbor")
                break
            next_pos = lowest_neigbor
            if lowest_neigbor in path:
                # This error may occur when using different neighborhood metrics in path finding and wavefront generation
                rospy.logerr("pathfinding failed! Stuck in a dead end!")
                break
            path.append(lowest_neigbor)
        return path

    

    def get_time_cost(self, current_time: float, index:int) -> float:
        driving_cost : float = 1 if index < 4 else 1.4
        return current_time + driving_cost


    def draw_timings(self, grid: np.ndarray, obstacles: np.ndarray, start: tuple[int, int], goal: tuple[int, int], path:list[tuple[int, int]]=[]) -> None:
        min_val = np.min(grid)
        max_val  = np.max(grid)
        image_matrix = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                val = grid[i, j]
                if obstacles[i][j] == 0:
                    image_matrix[i, j] = (0, 0, 0)  # black for obstacles
                elif val == -1:
                    image_matrix[i, j] = (255, 255, 255)  # white for non-visited spaces
                else:
                    blue_value = int(200 * (val - min_val) / (max_val - min_val)) + 55
                    image_matrix[i, j] = (255 - blue_value, 0, blue_value)  # red/blue depending on timing
        
        # path:
        for point in path:
            image_matrix[point[0], point[1]] = (0, 125 , 0)


        # start and goal:
        image_matrix[goal[0], goal[1]] = (255, 0, 0)
        image_matrix[start[0], start[1]] = (0, 255, 0)

        image_msg : Image = self.cv_bridge.cv2_to_imgmsg(image_matrix, encoding="rgb8")
        rospy.loginfo("published a new image")
        self.timing_pub.publish(image_msg)
        return None
    


    


    def draw_path_on_image_colored(self, img, path, start, goal):
        path_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.circle(path_img, (start[1], start[0]), self.point_size, (0, 255, 0), -1)
        cv2.circle(path_img, (goal[1], goal[0]), self.point_size, (0, 0, 255), -1)
        for pos in path:
            path_img[pos[0], pos[1]] = [255, 0, 0]  # Set the pixel to blue
        return path_img


if __name__ == '__main__':
    try:
        wavefront_expansion_node = WavefrontExpansionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
