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

        self.occupied_from : np.ndarray | None = None
        self.occupied_until: np.ndarray | None = None



    def image_callback(self, img_msg):
        img = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding="mono8")
        grid = self.process_image(img)

        start_position : tuple[int, int] = (45, 25)
        goal_position : tuple[int, int] = (120, 120)

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

        queue : list[tuple[int, int]] = [start_pos]
        rows = len(static_obstacles)
        cols = len(static_obstacles[0])

        direct_neighbors : list[tuple[int, int]] = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        diagonal_neighbors: list[tuple[int, int]] = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
        neigbors: list[tuple[int, int]] = direct_neighbors #+ diagonal_neighbors

        timings : np.ndarray = np.zeros((rows, cols))
        iterations : int = 0

        while queue:
            iterations += 1
            current_element : tuple[int, int] = queue.pop(0)
            current_time : float = timings[current_element[0], current_element[1]]

            if iterations % 10000 == 0:
                rospy.loginfo(f"{iterations} done!")

            if iterations > 500000:
                rospy.loginfo("breaking cuz took too long")
                break

            if current_element == goal_pos:
                rospy.loginfo(f"Reached the goal after {iterations} iterations")
                break

            for elem, (x_neighbor, y_neighbor) in enumerate(neigbors):
                x, y = current_element[0] + x_neighbor, current_element[1] + y_neighbor
                if 0 <= x < rows and 0 <= y < cols and static_obstacles[x][y] != 0: 
                

                    driving_cost : float = self.get_time_cost(current_time, elem)
                    #rospy.loginfo(f"timing: {timings[x, y]}, cost: {driving_cost}!")

                    if 0 < timings[x, y] < driving_cost: # there is a better route that leads to this point
                        continue

                    timings[x, y] = driving_cost
                    queue.append((x, y))

        
        end_time = time.time()
        elapsed_time = end_time - start_time
        rospy.loginfo(f"It took {elapsed_time} s.")
        self.draw_timings(timings, static_obstacles, start_pos, goal_pos)
        return None
    

    def get_time_cost(self, current_time: float, index:int) -> float:
        driving_cost : float = 1 if index < 4 else 1.5
        return current_time + driving_cost


    def draw_timings(self, grid: np.ndarray, obstacles: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> None:
        min_val = np.min(grid)
        max_val  = np.max(grid)
        image_matrix = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                val = grid[i, j]
                if obstacles[i][j] == 0:
                    image_matrix[i, j] = (0, 0, 0)  # Schwarz für -1
                elif val == 0:
                    image_matrix[i, j] = (255, 255, 255)  # Weiß für 0
                else:
                    blue_value = int(200 * (val - min_val) / (max_val - min_val)) + 55
                    image_matrix[i, j] = (255 - blue_value, 0, blue_value)  # Blau, wobei niedrigere Werte heller sind
        image_matrix[goal[0], goal[1]] = (255, 0, 0)
        image_matrix[start[0], start[1]] = (0, 255, 0)

        image_msg : Image = self.cv_bridge.cv2_to_imgmsg(image_matrix, encoding="rgb8")
        rospy.loginfo("publishing an image! :D")
        self.timing_pub.publish(image_msg)



    def wavefront_expansion_copy(self, grid, start, goal):
        rows = len(grid)
        cols = len(grid[0])

        # Initialisierung der Warteschlange mit dem Startpunkt
        queue = [start]
        visited = set()

        # Vorherige Zellen-Verweisungen für die Rückverfolgung des Pfades
        previous = {}

        # Wavefront-Expansion
        while queue:
            current = queue.pop(0)
            visited.add(current)

            # Wenn das Ziel erreicht wurde, beende die Expansion
            if current == goal:
                break

            # Präferenz für 4er-Nachbarschaft, es sei denn, 8er-Nachbarschaft ist schneller
            preferred_neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            if (goal[0] - current[0]) * (goal[1] - current[1]) != 0:  # Wenn das Ziel diagonal zum aktuellen Punkt liegt
                preferred_neighbors += [(1, 1), (-1, 1), (1, -1), (-1, -1)]

            # Überprüfen der Nachbarn des aktuellen Punktes
            for dr, dc in preferred_neighbors:
                new_row, new_col = current[0] + dr, current[1] + dc

                # Überprüfen, ob der Nachbar innerhalb der Grenzen des Rasters liegt und nicht blockiert ist
                if 0 <= new_row < rows and 0 <= new_col < cols and grid[new_row][new_col] != 0:
                    neighbor = (new_row, new_col)
                    if neighbor not in visited and neighbor not in queue:
                        queue.append(neighbor)
                        previous[neighbor] = current

        # Rückverfolgung des PfadesTrueTrue
        path = []
        current = goal
        while current != start:
            path.append(current)
            current = previous[current]
        path.append(start)
        path.reverse()

        return path
    
    def draw_path_on_image(self, img, path):
        for pos in path:
            img[pos[0], pos[1]] = 127  # Set the pixel to blue

    def draw_path_on_image_colored(self, img, path, start, goal):
        path_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert to color image
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
