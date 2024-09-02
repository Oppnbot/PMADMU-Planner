#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-


# Bugfixes:
# todo: don't communicate via magic numbers, use publisher and subscribers

# Code Quality:
# todo:

# Additional Features:
# todo: fix error when all starting positions are also goal positions and close to eachother
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
import cv2
import time
import heapq
from pmadmu_planner.msg import PixelPos, GoalPose, Trajectory, FollowerFeedback
from pmadmu_planner.msg import Waypoint as WaypointMsg
from pmadmu_planner.srv import TransformPixelToWorld, TransformPixelToWorldResponse, TransformWorldToPixel, TransformWorldToPixelResponse
from geometry_msgs.msg import Pose
from nav_msgs.msg import OccupancyGrid

from pmadmu_planner.cfg import PathFinderConfig

class Waypoint():
    def __init__(self, pixel_pos : tuple[int, int], occupied_from: float, occupied_until : float = float('inf'), world_pos : Pose|None = None, previous_waypoint:Waypoint|None = None):
        """
        Each grid cell the robot passes counts as a Waypoint. Multiple Waypoints make up the robots path.

        This Class contains the same data as the Waypoint msg but is needed for recursive operations and also offers an override for comparision etc, needed to execute the planning algorithm.
        
        :params world_pos: the (x, y) position in world coordinates [m]; used to navigate the robot
        :params pixel_pos: the (x, y) position in pixel coordinates [px]; used to find a path
        :params occupied_from: time when waypoint first becomes occupied, making it unavailable for others
        :params occupied_until: time when waypoint becomes free, making it available for other robots [s]
        """
        self.world_pos : Pose | None = world_pos # the (x, y) position in world coordinates [m]; used to navigate the robot
        self.pixel_pos : tuple[int, int] = pixel_pos            # the (x, y) position in pixel coordinates [px]; used to find a path
        self.occupied_from: float = occupied_from               # time when waypoint first becomes occupied, making it unavailable for other robots [s]
        self.occupied_until: float = occupied_until              # time when waypoint becomes free, making it available for other robots [s]
        self.previous_waypoint : Waypoint | None = previous_waypoint
        self.generation : int = 0

    def __eq__(self, __value: Waypoint) -> bool:
        return self.pixel_pos == __value.pixel_pos

    def __lt__(self, other : Waypoint):
        return self.occupied_from < other.occupied_from
    
    def convert_to_msg(self) -> WaypointMsg:
        waypoint_msg : WaypointMsg = WaypointMsg()
        waypoint_msg.occupied_from = self.occupied_from
        waypoint_msg.occupied_until = self.occupied_until
        
        if self.world_pos is not None:
            waypoint_msg.world_position = self.world_pos

        pixel_pos : PixelPos = PixelPos()
        pixel_pos.x, pixel_pos.y = self.pixel_pos

        waypoint_msg.pixel_position = pixel_pos
        return waypoint_msg



class PathFinder:
    def __init__(self, robot_name : str = "unknown", robot_id : int = -1):
        # -------- CONFIG START --------
        #! you can change these parameters via rqt or the FollowerConfig.cfg file
        self.static_time_tolerance: float = 4.0     # [s] estimation of how long the robot needs to travel between two grid spaces. will add a static length to the "snakes"
        self.dynamic_time_tolerance : float = 1.5   # [s] estimation of motion uncertainty. lets "snakes" grow over time

        self.allow_straights : bool = True   # allows the following movements: 0°, 90°, 180°, 380° 
        self.allow_diagonals : bool = True   # allows the following movements: 45°, 135°, 225°, 315°
        self.allow_knight_moves: bool = False # allows the following movements: 26°, 63°, 116°, 153°, 206°, 243°, 296°, 333° (like a knight in chess)

        self.check_dynamic_obstacles : bool = True
        self.dynamic_visualization : bool = False # publishes timing map after every step, very expensive
        
        self.speed : float = 0.5
        # -------- CONFIG END --------
        self.kernel_size : int = 3 #!kernel size -> defines the safety margins for dynamic and static obstacles; grid_size * kernel_size = robot_size
        

        self.robot_name : str = robot_name
        self.robot_id : int = robot_id
        self.robot_pose : Pose | None = None
        self.static_obstacles : OccupancyGrid | None = None
        
        rospy.Subscriber(f'/{robot_name}/robot_pose', Pose, self.update_pose)
        rospy.Subscriber(f'/{robot_name}/mir_pose_simple', Pose, self.update_pose)
        rospy.Subscriber('/pmadmu_planner/static_obstacles', OccupancyGrid, self.read_static_obstacles)
        self.status_publisher = rospy.Publisher('/pmadmu_planner/follower_status', FollowerFeedback, queue_size=10, latch=True)
        #self.trajectory_publisher : rospy.Publisher = rospy.Publisher('pmadmu_planner/trajectory', Trajectory, queue_size=10, latch=True)
        return None
    


    def config_change(self, config, level):
        self.static_time_tolerance = config.static_time_tolerance
        self.dynamic_time_tolerance = config.dynamic_time_tolerance
        self.allow_straights = config.allow_straights
        self.allow_diagonals = config.allow_diagonals
        self.allow_knight_moves = config.allow_knight_moves
        self.check_dynamic_obstacles = config.check_dynamic_obstacles
        self.dynamic_visualization = config.dynamic_visualization
        self.speed = config.speed
        rospy.loginfo(f"[Planner {self.robot_name}] Applied config changes")
        return config



    def update_pose(self, pose: Pose) -> None:
        self.robot_pose = pose
        return None
    
    def read_static_obstacles(self, static_obstacles: OccupancyGrid) -> None:
        self.static_obstacles = static_obstacles
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
                rospy.logwarn(f"[Planner {self.robot_name}] min occupation time is infinite")
            if max_occupied_until == 0:
                rospy.logwarn(f"[Planner {self.robot_name}] max occupation time is 0")
            new_waypoint : Waypoint = Waypoint(position, min_occupied_from, max_occupied_until)
            bloated_path.append(new_waypoint)
        return bloated_path
    


    def search_path(self, static_obstacles: np.ndarray, goal_pos: GoalPose, dynamic_obstacles: list[Trajectory] = []) -> Trajectory | None:
        iterations : int = 0
        while self.robot_pose is None:
            rospy.logwarn(f"[Planner {self.robot_name}] failed to plan path, waiting for robot position update. Retrying...")
            rospy.sleep(0.25)
            iterations += 1
        rospy.loginfo(f"[Planner {self.robot_name}] Starting Trajectory Search")

        start_time = time.time()

        # Get current position and transform it to pixel space
        transform_world_to_pixel = rospy.ServiceProxy('/pmadmu_planner/world_to_pixel', TransformWorldToPixel)

        w2p_response : TransformWorldToPixelResponse = transform_world_to_pixel([self.robot_pose.position.x], [self.robot_pose.position.y])
        if len(w2p_response.x_pixel) == 0 or len(w2p_response.y_pixel) == 0:
            rospy.logwarn(f"[Planner {self.robot_name}] failed to plan path, since robot starting position couldn't be converted to pixel space.")
            return None
        robot_start_pixel_pos : tuple[int, int] = (w2p_response.x_pixel[0], w2p_response.y_pixel[0])
        start_waypoint : Waypoint = Waypoint(robot_start_pixel_pos, 0.0)
        
        #cv2.imshow(f"static obst {self.robot_name}", static_obstacles)
        #cv2.waitKey()
        static_obstacles[robot_start_pixel_pos[0], robot_start_pixel_pos[1]] = 1 #? this may prevent robots from being stuck in an obstacles inflation zone

        # Get goal position and transform it to pixel space
        w2p_response : TransformWorldToPixelResponse = transform_world_to_pixel([goal_pos.goal.position.x], [goal_pos.goal.position.y])
        if len(w2p_response.x_pixel) == 0 or len(w2p_response.y_pixel) == 0:
            rospy.logwarn(f"[Planner {self.robot_name}] failed to plan path, since its goal position couldn't be converted to pixel space.")
            return None
        robot_goal_pixel_pos : tuple[int, int] = (w2p_response.x_pixel[0], w2p_response.y_pixel[0])
        goal_waypoint : Waypoint = Waypoint(robot_goal_pixel_pos, float('inf'))

        occupied_positions : dict[tuple[float, float], list[Waypoint]] = {}

        
        bloating_time_start = time.time()
        bloated_dynamic_obstacles : list[Waypoint] = []
        for dyn_obst in dynamic_obstacles:
            dynamic_obstacle : Trajectory = dyn_obst

            if dynamic_obstacle.occupied_positions is None:
                continue

            waypoints : list[Waypoint] = [Waypoint((waypoint_msg.pixel_position.x, waypoint_msg.pixel_position.y), waypoint_msg.occupied_from, waypoint_msg.occupied_until) for waypoint_msg in dynamic_obstacle.occupied_positions]
            bloated_dynamic_obstacles = self.bloat_path(waypoints)
            for waypoint in bloated_dynamic_obstacles:
                occupied_positions.setdefault(waypoint.pixel_pos, []).append(waypoint)
        bloating_time_done = time.time() 
        rospy.loginfo(f"[Planner {self.robot_name}] inflated paths. this took {bloating_time_done - bloating_time_start:.6f}s")

        heap: list[tuple[float, Waypoint]] = [(0, start_waypoint)]

        
        # Dilate the obstacles by ~1/2 of the robots size to avoid collisions
        dilation_time_start = time.time()
        kernel = np.ones((self.kernel_size, self.kernel_size),np.uint8) #type: ignore
        bloated_static_obstacles : np.ndarray = cv2.erode(static_obstacles, kernel) #-> erosion of free space = dilation of obstacles
        dilation_time_done = time.time()
        rospy.loginfo(f"[Planner {self.robot_name}] Dilated map. This took {dilation_time_done-dilation_time_start:.6f}s")

        rows: int = bloated_static_obstacles.shape[0]
        cols: int = bloated_static_obstacles.shape[1]
        
        timings: np.ndarray = np.full((rows, cols), -1.0)
        timings[robot_start_pixel_pos] = 0.0

        # Adjust starting positions
        if start_waypoint in bloated_dynamic_obstacles:
            timings[robot_start_pixel_pos] = 100
            rospy.logerr(f"[Planner {self.robot_name}]: Other Robot driving through my starting pos!")


        
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

        grid_to_real_conversion = 1
        if self.static_obstacles is not None:
            grid_to_real_conversion = self.static_obstacles.info.resolution
        neighbor_costs : list[float] = [(np.hypot(x, y) * grid_to_real_conversion) / (self.speed ) for x, y in neighbors]

        loop_time_start = time.time()
        iterations : int = 0
        max_iterations : int = static_obstacles.size
        while heap:
            iterations += 1
            current_cost, current_waypoint = heapq.heappop(heap)
            if iterations % 10_000 == 0:
                rospy.loginfo(f"[Planner {self.robot_name}] {iterations} iterations done!")
            if iterations > max_iterations:
                rospy.logwarn(f"[Planner {self.robot_name}] breaking because algorithm reached max iterations {max_iterations}")
                return None
            
            #if self.dynamic_visualization:
            #    fb_visualizer.draw_timings(timings, bloated_static_obstacles, start_waypoint.pixel_pos, goal_waypoint.pixel_pos, dynamic_obstacles=dynamic_obstacles, sleep=None)

            # CHECK CURRENT POSITION FOR COLLISION
            # if another robot is blocking the path, the robot waits for the path to become free. this may lead to collision with other robots that try to drive through this waiting position
            # the proposed solution is to check for other robots and if a conflict is found, the waiting position will be changed to its parent position.
            if self.check_dynamic_obstacles and (current_waypoint.pixel_pos in occupied_positions.keys()):
                is_occupied : bool = False
                for waypoint in occupied_positions[current_waypoint.pixel_pos]:
                    current_wp_occ_until = current_waypoint.occupied_from * self.dynamic_time_tolerance + self.static_time_tolerance
                    if current_waypoint == goal_waypoint:
                        current_wp_occ_until = float('inf')
                    # does the occupation timeframe overlap with the occupation of a higher prio robot?
                    if current_waypoint.occupied_from < waypoint.occupied_from < current_wp_occ_until or current_waypoint.occupied_from < waypoint.occupied_until < current_wp_occ_until:
                        #rospy.logwarn(f"robot {self.robot_name} would collide at position {current_waypoint.pixel_pos} after {current_cost}s while waiting. it is occupied between {waypoint.occupied_from}s -> {waypoint.occupied_until}s ")
                        is_occupied = True
                        if current_waypoint == start_waypoint:
                            rospy.logerr(f"[Planner {self.robot_name}] can't adjust waiting position since robot already waiting at its starting pos")
                            return None
                        if current_waypoint.previous_waypoint is not None:
                            #rospy.loginfo(f"will wait at {current_waypoint.previous_waypoint.pixel_pos} until {waypoint.occupied_until}")
                            if waypoint.occupied_until == float('inf'):
                                continue
                            timings[current_waypoint.pixel_pos[0], current_waypoint.pixel_pos[1]] = -1
                            heapq.heappush(heap, (waypoint.occupied_until, current_waypoint.previous_waypoint))
                        else:
                            rospy.logwarn(f"[Planner {self.robot_name}] will collide at current position {current_waypoint.pixel_pos} but previous waypoint is none!")
                        break
                if is_occupied:
                    continue

            if current_waypoint == goal_waypoint:
                rospy.loginfo(f"[Planner {self.robot_name}] Reached the goal after {iterations} iterations")
                goal_waypoint = current_waypoint
                break

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
                rospy.logerr(f"[Planner {self.robot_name}] stopping because heap queue is empty")
                #feedback : FollowerFeedback = FollowerFeedback()
                #feedback.robot_id = self.id
                #feedback.status = feedback.PLANNING_FAILED
                #self.status_publisher.publish(feedback)
                return None
        
        rospy.loginfo(f"[Planner {self.robot_name}] stopped after a total of {iterations} iterations")
        loop_end_time = time.time()
        rospy.loginfo(f"[Planner {self.robot_name}] planned path! Dijkstras main loop took {loop_end_time-loop_time_start:.6f}s")
        

        #* --- Reconstruct Path ---
        pathfind_start_time = time.time()
        waypoints : list[Waypoint] = []
        current_waypoint : Waypoint | None = goal_waypoint
        while current_waypoint:
            if current_waypoint.previous_waypoint is not None:
                current_waypoint.previous_waypoint.occupied_until = (current_waypoint.occupied_from + 1) * self.dynamic_time_tolerance + self.static_time_tolerance
            waypoints.append(current_waypoint)
            current_waypoint = current_waypoint.previous_waypoint
            #if self.dynamic_visualization:
            #    fb_visualizer.draw_timings(timings, bloated_static_obstacles, start_waypoint.pixel_pos, goal_waypoint.pixel_pos, dynamic_obstacles=dynamic_obstacles, sleep=None)
        waypoints.reverse()

        bloated_waypoints : list[Waypoint] = self.bloat_path(waypoints)

        pathfind_done_time = time.time()

        expected_waypoints : int = max(abs(robot_goal_pixel_pos[0]-robot_start_pixel_pos[0]), abs(robot_goal_pixel_pos[1]-robot_start_pixel_pos[1]))
        if len(waypoints) < expected_waypoints:
            rospy.logwarn(f"[Planner {self.robot_name}] Path seems to be too short, this may be a result of an invalid path. got {len(waypoints)}, expected {expected_waypoints}; Discarding solution and trying to replan...")
            return None
        rospy.loginfo(f"[Planner {self.robot_name}] Found a path. This took {pathfind_done_time-pathfind_start_time:.6f}s")
        rospy.loginfo(f"[Planner {self.robot_name}] Shortest path consists of {len(waypoints)} nodes with a cost of {timings[robot_goal_pixel_pos[0], robot_goal_pixel_pos[1]]}")

        # Transform Path from Pixel-Space to World-Space for visualization and path following
        trafo_start_time = time.time()

        transform_pixel_to_world = rospy.ServiceProxy('/pmadmu_planner/pixel_to_world', TransformPixelToWorld)
        pixel_positions_x : list[int] = [waypoint.pixel_pos[0] for waypoint in waypoints]
        pixel_positions_y : list[int] = [waypoint.pixel_pos[1] for waypoint in waypoints]
        response : TransformPixelToWorldResponse = transform_pixel_to_world(pixel_positions_x, pixel_positions_y)
        for index, waypoint in enumerate(waypoints):
            #rospy.logwarn(f"world_pos {response.x_world[index]} / {response.y_world[index]}")
            if waypoint == goal_waypoint:
                waypoint.world_pos = goal_pos.goal
                continue
            pose : Pose = Pose()
            pose.position.x = response.x_world[index]
            pose.position.y = response.y_world[index]
            pose.position.z = 0.0
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            pose.orientation.w = 0.0
            waypoint.world_pos = pose

        pixel_positions_x : list[int] = [waypoint.pixel_pos[0] for waypoint in bloated_waypoints]
        pixel_positions_y : list[int] = [waypoint.pixel_pos[1] for waypoint in bloated_waypoints]
        response : TransformPixelToWorldResponse = transform_pixel_to_world(pixel_positions_x, pixel_positions_y)
        for index, waypoint in enumerate(bloated_waypoints):
            if waypoint == goal_waypoint:
                waypoint.world_pos = goal_pos.goal
                continue
            pose : Pose = Pose()
            pose.position.x = response.x_world[index]
            pose.position.y = response.y_world[index]
            pose.position.z = 0.0
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            pose.orientation.w = 0.0
            waypoint.world_pos = pose
        

        trafo_end_time = time.time()
        rospy.loginfo(f"[Planner {self.robot_name}] Transformed pixel data to world coordinates. This took {trafo_end_time-trafo_start_time:.6f}s")
        
        # Visualization
        #fb_visualizer.draw_timings(timings, bloated_static_obstacles, start_pos, goal_pos, trajectory_data.waypoints)
        
        trajectory : Trajectory = Trajectory()
        trajectory.robot_name = self.robot_name
        trajectory.planner_id = self.robot_id
        trajectory.goal_waypoint = goal_waypoint.convert_to_msg()
        trajectory.start_waypoint = start_waypoint.convert_to_msg()
        path_msgs : list[WaypointMsg] = [waypoint.convert_to_msg() for waypoint in waypoints]
        occupied_positions_msgs : list[WaypointMsg] = [waypoint.convert_to_msg() for waypoint in bloated_waypoints]
        trajectory.path = path_msgs
        trajectory.occupied_positions = occupied_positions_msgs
        #self.trajectory_publisher.publish(trajectory)

        end_time = time.time()
        rospy.loginfo(f"[Planner {self.robot_name}] Done! Took {end_time-start_time:.6f}s in total for this planner.")
        rospy.loginfo(f"[Planner {self.robot_name}] - - - - - - - - -")
        return trajectory