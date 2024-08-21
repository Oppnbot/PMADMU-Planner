#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-


# Bugfixes:
# todo: start/stop positioning are not scaling
# todo: don't communicate via magic numbers, use publisher and subscribers


# Code Quality:
# todo: replace waypoints by ros messages
# todo: distribute code to multiple nodes

# Additional Features:
# todo: option for wider paths to fit robot width (+map scaling)
# todo: assign new priorities when there is no viable path
# todo: assign new priorities when a path took too long (?)
# todo: implement path following
# todo: distribute calculation to multiple robots for O{(n-1)!} instead of O{n!}

# Other:
# todo: skip visited nodes ?
# todo: test in different scenarios
# todo: adjust to the robots real dynamics (?)
# todo: improve overall computing performance
# todo: improve occupation time calculations with safety margin etc

from __future__ import annotations
import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
import heapq
from commons import TrajectoryData, Waypoint
from visualization import fb_visualizer
#from formation_builder.srv import transformation
from formation_builder.srv import TransformPixelToWorld, TransformPixelToWorldResponse


class Spawner:
    def __init__(self)->None:
        # -------- CONFIG START --------
        self.path_finder_count : int = 4
        #self.starting_positions : list[tuple[int, int]] =  [(1, 1), (3, 1), (5, 1), (7, 1)]
        #self.goal_positions: list[tuple[int, int]] =       [(70, 38), (62, 17), (60, 39), (52, 12)]
        

        #* for 1.4m
        #self.starting_positions : list[tuple[int, int]] =  [(20, 20), (35, 48), (23, 19), (25, 45)]
        #self.goal_positions: list[tuple[int, int]] =       [(70, 38), (53, 11), (60, 39), (55, 13)]


        #* for 1.0m
        self.starting_positions : list[tuple[int, int]] =  [(28, 28), (49, 67), (32, 26), (38, 68)]
        self.goal_positions: list[tuple[int, int]] =       [(98, 53), (74, 15), (84, 55), (78, 18)]

        grid_size = 0.3 #m #! via publisher please
        robot_size = 1.0 #m #! config file

        factor : float = robot_size / grid_size

        self.starting_positions  : list[tuple[int, int]] =  [(int(np.round(x * factor)), int(np.round(y * factor))) for x, y in self.starting_positions]
        self.goal_positions  : list[tuple[int, int]] =      [(int(np.round(x * factor)), int(np.round(y * factor))) for x, y in self.goal_positions]
       




        # --------- CONFIG END ---------
        rospy.init_node('mapf')
        

        self.path_finders : list[WavefrontExpansionNode] = []
        self.cv_bridge : CvBridge = CvBridge()
        self.occupation: dict[tuple[int, int], list[tuple[float, float]]] = {} #key: x- and y-val of the gridcell. value: list of timings with "occupied from" and "occupied until"
        for i in range(self.path_finder_count):
            self.path_finders.append(WavefrontExpansionNode(i))
        

        rospy.loginfo("waiting for transformation service...")
        rospy.wait_for_service('pixel_to_world')
        rospy.loginfo("transformation service is running!")
        
        rospy.Subscriber('/formation_builder/map', Image, self.map_callback)

        return None
    


    

    def map_callback(self, img_msg : Image) -> None:
        grid : np.ndarray = self.map_to_grid(img_msg)
        trajectories : list[TrajectoryData] = []

        fb_visualizer.clear_image(grid.shape[0], grid.shape[1])
        fb_visualizer.draw_obstacles(grid)

        start_time = time.time()

        # fill trajectory data with starting positions to avoid collisions
        for i in range(len(self.path_finders)):
            if i >= len(self.goal_positions):
                break
            starting_waypoints : list[Waypoint] = []
            kernel_size :int = 3#! make this a parameter
            half_kernel_size : int = (kernel_size - 1) // 2
            starting_waypoint : Waypoint = Waypoint(self.starting_positions[i], 0, float('inf'))
            #todo: use bloating function instead?
            starting_waypoints.append(starting_waypoint)
            for x in range(-half_kernel_size, half_kernel_size+1):
                for y in range(-half_kernel_size, half_kernel_size+1):
                    if x == 0 and y == 0:
                        continue
                    position : tuple[int, int] = (self.starting_positions[i][0] + x, self.starting_positions[i][1] + y)
                    starting_waypoints.append(Waypoint(position, 0, float('inf'), previous_waypoint=starting_waypoint))
            trajectories.append(TrajectoryData(i, starting_waypoints))


        for index, path_finder in enumerate(self.path_finders):
            if index >= len(self.starting_positions) or index >= len(self.goal_positions):
                rospy.logwarn("there are not enough starting/goal positions for the chosen ammount of planners.")
                break
            trajectories.pop(0) # remove the first element which should be the starting pos
            trajectory : TrajectoryData = path_finder.path_finder(grid, self.starting_positions[index], self.goal_positions[index], trajectories)
            #trajectory : TrajectoryData = path_finder.path_finder(grid, self.starting_positions[index], self.goal_positions[index], self.occupation)
            trajectories.append(trajectory)

            #self.update_occupation_dict(trajectory)

            fb_visualizer.draw_path(trajectory, self.path_finder_count)
            fb_visualizer.draw_start_and_goal(trajectory, self.path_finder_count)
            fb_visualizer.publish_image()
        end_time = time.time()

        rospy.loginfo(f"------------------ Done! ({end_time-start_time:.3f}s) ------------------ ")
        fb_visualizer.show_live_path(trajectories)
        return None



    def map_to_grid(self, map : Image) -> np.ndarray:
        img = self.cv_bridge.imgmsg_to_cv2(map, desired_encoding="mono8")
        grid : np.ndarray = np.array(img) // 255
        return grid
    


class WavefrontExpansionNode:
    def __init__(self, planner_id : int = 0):
        # -------- CONFIG START --------
        self.allow_straights : bool = True   # allows the following movements: 0°, 90°, 180°, 380° 
        self.allow_diagonals : bool = True   # allows the following movements: 45°, 135°, 225°, 315°
        self.allow_knight_moves: bool = False # allows the following movements: 26°, 63°, 116°, 153°, 206°, 243°, 296°, 333° (like a knight in chess)

        self.check_dynamic_obstacles : bool = True
        self.dynamic_visualization : bool = False # publishes timing map after every step, very expensive
        self.kernel_size : int = 3 #!kernel size -> defines the safety margins for dynamic and static obstacles; grid_size * kernel_size = robot_size
        # -------- CONFIG END --------
        
        self.id: int = planner_id
        return None
    

    def bloat_path(self, waypoints : list[Waypoint]) -> list[Waypoint]:
        bloated_path : list[Waypoint] = []
        occupied_positions : dict[tuple[int, int], list[Waypoint]] = {}
        for waypoint in waypoints:
            if waypoint.pixel_pos not in occupied_positions.keys():
                occupied_positions[waypoint.pixel_pos] = []
            occupied_positions[waypoint.pixel_pos].append(waypoint)
            bloat_width : int = (self.kernel_size - 1) // 2
            for x in range(waypoint.pixel_pos[0] - bloat_width, waypoint.pixel_pos[0] + bloat_width + 1):
                for y in range(waypoint.pixel_pos[1] - bloat_width, waypoint.pixel_pos[1] + bloat_width +1):
                    if (x, y) not in occupied_positions.keys():
                        occupied_positions[x, y] = []
                    new_waypoint : Waypoint = Waypoint((x,y), waypoint.occupied_from, waypoint.occupied_until)
                    occupied_positions[(x, y)].append(new_waypoint)
        for position, waypoints in occupied_positions.items():
            if not waypoints:
                continue
            min_occupied_from : float = float('inf')
            max_occupied_until : float = 0
            for waypoint in waypoints:
                if waypoint.occupied_from < min_occupied_from:
                    min_occupied_from = waypoint.occupied_from
                if waypoint.occupied_until > max_occupied_until:
                    max_occupied_until = waypoint.occupied_until
            if min_occupied_from == float('inf'):
                rospy.logwarn("min occupation time is infinite")
            if max_occupied_until == 0:
                rospy.logwarn("max occupation time is 0")
            new_waypoint : Waypoint = Waypoint(position, min_occupied_from, max_occupied_until)
            bloated_path.append(new_waypoint)
        return bloated_path



    def path_finder(self, static_obstacles: np.ndarray, start_pos: tuple[int, int], goal_pos: tuple[int, int], dynamic_obstacles: list[TrajectoryData] = []) -> TrajectoryData:
        rospy.loginfo(f"Planner {self.id} Starting Trajectory Search")

        start_time = time.time()

        start_waypoint : Waypoint = Waypoint(start_pos, 0.0)
        goal_waypoint : Waypoint = Waypoint(goal_pos, float('inf'))

        occupied_positions : dict[tuple[float, float], list[Waypoint]] = {}

        
        bloating_time_start = time.time()
        for dynamic_obstacle in dynamic_obstacles:
            bloated_dynamic_obstacles : list[Waypoint] = self.bloat_path(dynamic_obstacle.waypoints)
            for waypoint in bloated_dynamic_obstacles:
                occupied_positions.setdefault(waypoint.pixel_pos, []).append(waypoint)
        bloating_time_done = time.time() 
        rospy.loginfo(f"bloated paths. this took {bloating_time_done - bloating_time_start:.6f}s")

        heap: list[tuple[float, Waypoint]] = [(0, start_waypoint)]

        
        # Dilate the obstacles by ~1/2 of the robots size to avoid collisions
        dilation_time_start = time.time()
        kernel = np.ones((self.kernel_size, self.kernel_size),np.uint8) #type: ignore
        bloated_static_obstacles : np.ndarray = cv2.erode(static_obstacles, kernel) #-> erosion of free space = dilation of obstacles
        dilation_time_done = time.time()
        rospy.loginfo(f"dilated map. this took {dilation_time_done-dilation_time_start:.6f}s")

        rows: int = bloated_static_obstacles.shape[0]
        cols: int = bloated_static_obstacles.shape[1]
        
        timings: np.ndarray = np.full((rows, cols), -1.0)
        timings[start_pos] = 0.0

        
        direct_neighbors: list[tuple[int, int]] = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        diagonal_neighbors: list[tuple[int, int]] = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
        knight_neighbors: list[tuple[int, int]] = [(2, 1), (-2, 1), (2, -1), (-2, -1), (1, 2), (-1, 2), (1, -2), (-1, -2)]
        neighbors: list[tuple[int, int]] = []
        if self.allow_straights:
            neighbors += direct_neighbors
        if self.allow_diagonals:
            neighbors += diagonal_neighbors
        if self.allow_knight_moves:
            neighbors += knight_neighbors
        neighbor_costs : list[float] = [np.hypot(x, y) for x, y in neighbors]

        loop_time_start = time.time()
        iterations : int = 0
        while heap:
            iterations += 1
            current_cost, current_waypoint = heapq.heappop(heap)
            if iterations % 10_000 == 0:
                rospy.loginfo(f"planner {self.id}: {iterations} iterations done!")
            if iterations > 500_000:
                rospy.logwarn(f"planner {self.id}: breaking because algorithm reached max iterations")
                break
            if current_waypoint.pixel_pos == goal_pos:
                rospy.loginfo(f"planner {self.id}: Reached the goal after {iterations} iterations")
                goal_waypoint = current_waypoint
                break
            if self.dynamic_visualization:
                fb_visualizer.draw_timings(timings, bloated_static_obstacles, start_pos, goal_pos, dynamic_obstacles=dynamic_obstacles, sleep=None)

            # CHECK CURRENT POSITION FOR COLLISION
            # if another robot is blocking the path, the robot waits for the path to become free. this may lead to collision with other robots that try to drive through this waiting position
            # the proposed solution is to check for other robots and if a conflict is found, the waiting position will be changed to its parent position.
            if self.check_dynamic_obstacles and (current_waypoint.pixel_pos in occupied_positions.keys()):
                is_occupied = False
                for waypoint in occupied_positions[current_waypoint.pixel_pos]:
                    if current_waypoint.occupied_from <= waypoint.occupied_from:
                        #rospy.logwarn(f"robot {self.id} would collide at position {current_waypoint.pixel_pos} after {current_cost}s while waiting. it is occupied between {waypoint.occupied_from}s -> {waypoint.occupied_until}s ")
                        is_occupied = True
                        if current_waypoint.previous_waypoint is not None:
                            #rospy.loginfo(f"will wait at {current_waypoint.previous_waypoint.pixel_pos} until {waypoint.occupied_until}")
                            if waypoint.occupied_until == float('inf'):
                                continue
                            timings[current_waypoint.pixel_pos[0], current_waypoint.pixel_pos[1]] = -1
                            heapq.heappush(heap, (waypoint.occupied_until, current_waypoint.previous_waypoint)) # we have to add a small number because of floating point issues
                        else:
                            rospy.logwarn(f"will collide at current position {current_waypoint.pixel_pos} but previous waypoint is none!")
                        break
                if is_occupied:
                    continue

            # GRAPH EXPANSION
            # look at the neighbors; expand if possible. The neighbor has to contain free space, meaning no static/dynamic obstacle
            # we also have to assign a cost depending on the time needed to reach this node. this is the parent cost + an additional driving cost
            for index, (x_neighbor, y_neighbor) in enumerate(neighbors):
                x, y = current_waypoint.pixel_pos[0] + x_neighbor, current_waypoint.pixel_pos[1] + y_neighbor
                if 0 <= x < rows and 0 <= y < cols and bloated_static_obstacles[x, y] != 0: # check for static obstacles / out of bounds
  
                    driving_cost = current_cost + neighbor_costs[index]
                    # DYNAMIC OBSTACLE CHECK
                    # if the neighbor node is currently occupied by another robot, wait at the current position. To do that we add the current position back into the heap
                    # but with an updated timing, that is equal to the time needed for the robot to free the position.
                    if self.check_dynamic_obstacles and (x, y) in occupied_positions.keys(): # check for dynamic obstacles
                        is_occupied : bool = False
                        for waypoint in occupied_positions[(x, y)]:
                            if waypoint.occupied_from <= driving_cost <= waypoint.occupied_until:
                                if waypoint.pixel_pos == goal_pos: # goalpos was found but occupied -> wait and stop searching
                                    heap = [(waypoint.occupied_until, current_waypoint)]
                                heapq.heappush(heap, (waypoint.occupied_until, current_waypoint))
                                is_occupied = True
                                break
                        if is_occupied:
                            continue

                    # EXPANSION
                    # cell contains free space -> add it to the heap with the calculated driving cost. since the heap works like a priority queue, it will automatically be sorted,
                    # so that we proceed with the earliest remaining node.
                    if driving_cost < timings[x, y] or timings[x, y] < 0:
                        timings[x, y] = driving_cost
                        new_waypoint : Waypoint = Waypoint((x,y), driving_cost, previous_waypoint=current_waypoint)
                        heapq.heappush(heap, (driving_cost, new_waypoint))
            if not heap:
                rospy.loginfo(f"planner {self.id}s stopping because heap queue is empty")
        
        rospy.loginfo(f"planner {self.id}: stopped after a total of {iterations} iterations")
        loop_end_time = time.time()
        rospy.loginfo(f"planner {self.id}: planned path! Dijkstras main loop took {loop_end_time-loop_time_start:.6f}s")
        

        #* --- Reconstruct Path ---
        pathfind_start_time = time.time()
        waypoints : list[Waypoint] = []
        current_waypoint : Waypoint | None = goal_waypoint
        while current_waypoint:
            if current_waypoint.previous_waypoint is not None:
                current_waypoint.previous_waypoint.occupied_until = (current_waypoint.occupied_from + 1)* 1.1 + 1.0 # todo: define different metrics here #*1.3+1.0
            waypoints.append(current_waypoint)
            current_waypoint = current_waypoint.previous_waypoint
            if self.dynamic_visualization:
                fb_visualizer.draw_timings(timings, bloated_static_obstacles, start_pos, goal_pos, waypoints)
        waypoints.reverse()

        bloated_waypoints : list[Waypoint] = self.bloat_path(waypoints)
        trajectory_data : TrajectoryData = TrajectoryData(self.id, waypoints)
        trajectory_data.waypoints = bloated_waypoints
        pathfind_done_time = time.time()
        rospy.loginfo(f"planner {self.id}: found a path. This took {pathfind_done_time-pathfind_start_time:.6f}s")
        



        rospy.loginfo(f"planner {self.id}: shortest path consists of {len(waypoints)} nodes with a cost of {timings[goal_pos[0], goal_pos[1]]}")

        # Transform Path from Pixel-Space to World-Space for visualization and path following
        trafo_start_time = time.time()
        transform_pixel_to_world = rospy.ServiceProxy('pixel_to_world', TransformPixelToWorld)
        pixel_positions_x : list[int] = [waypoint.pixel_pos[0] for waypoint in trajectory_data.waypoints]
        pixel_positions_y : list[int] = [waypoint.pixel_pos[1] for waypoint in trajectory_data.waypoints]

        response : TransformPixelToWorldResponse = transform_pixel_to_world(pixel_positions_x, pixel_positions_y)
        for i in range(len(response.x_world)):
            trajectory_data.waypoints[i].world_pos = (response.x_world[i], response.y_world[i])

        trafo_end_time = time.time()
        rospy.loginfo(f"planner {self.id}: Transformed pixel data to world coordinates. This took {trafo_end_time-trafo_start_time:.6f}s")
        
        # Visualization
        fb_visualizer.draw_timings(timings, bloated_static_obstacles, start_pos, goal_pos, trajectory_data.waypoints)

        end_time = time.time()
        rospy.loginfo(f"planner {self.id} done! Took {end_time-start_time:.6f}s in total for this planner.")
        rospy.loginfo("- - - - - - - - -")
        return trajectory_data


if __name__ == '__main__':
    spawner : Spawner = Spawner()
    rospy.spin()

