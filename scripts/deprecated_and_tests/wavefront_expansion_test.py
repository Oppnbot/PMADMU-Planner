#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


class WavefrontExpansionNode:
    def __init__(self):
        rospy.init_node('wavefront_expansion_node')
        rospy.Subscriber('/formation_builder/map', Image, self.image_callback)
        self.image_pub = rospy.Publisher("/path_image", Image, queue_size=1, latch=True)
        self.path_pub = rospy.Publisher("/path_output", Path, queue_size=10, latch=True)
        self.grid_resolution = 2.0
        self.cv_bridge = CvBridge()
        self.point_size : int = 2

    def image_callback(self, img_msg):
        img = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding="mono8")
        grid = self.process_image(img)

        start = (45, 25)
        goal = (120, 120)

        path = self.wavefront_expansion(grid, start, goal)
        rospy.loginfo(f"path {path}")
        path_img = self.draw_path_on_image_colored(img, path, start, goal)
        path_image_msg = self.cv_bridge.cv2_to_imgmsg(path_img, encoding="bgr8")
        self.image_pub.publish(path_image_msg)

    def process_image(self, img):
        # Umwandlung des Bildes in ein Rastergitter
        grid = np.array(img) // 255
        return grid.tolist()

    def wavefront_expansion(self, grid, start, goal):
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
