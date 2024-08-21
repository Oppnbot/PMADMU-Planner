#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-


from __future__ import annotations

import time
import rospy
import numpy as np
from visualization_msgs.msg import Marker
from sensor_msgs.msg import LaserScan
from laser_scanner import LaserScanner
from nav_msgs.msg import OccupancyGrid
import os
import csv
from pmadmu_planner.srv import TransformWorldToPixel, TransformWorldToPixelResponse
from geometry_msgs.msg import Twist, PoseStamped, Point, Pose
from pmadmu_planner.msg import Trajectory, Trajectories, Waypoint, FollowerFeedback
from tf.transformations import euler_from_quaternion


class PathFollower:
    def __init__(self, robot_id:int) -> None:
        #rospy.init_node(f'path_follower_{robot_id}')
        rospy.loginfo(f"[Follower {robot_id}] Initializing!")
        self.robot_id : int = robot_id
        self.trajectory : Trajectory | None = None
        self.trajectory_start_time : float = 0
        self.robot_pose : Pose | None = None
        self.costmap : OccupancyGrid | None = None

        # ---- Config Zone ----
        self.lookahead_distance : float = 1.2   # [m] higher distance -> smoother curves when driving but might leave path.
        self.lookahead_time : float = 10        # [s] higher value -> earlier replanning when a collision with a previously unknown obstacle is inbound. please note that it may take a few seconds to update the obstacles, so >= 5s is advised
        self.max_linear_speed : float = 1.0  #0.15   # [m/s] max driving speed
        self.max_angular_speed : float = 1.5  #0.2   # [rad/s] max rotation speed
        
        self.goal_tolerance : float = 0.05 # [m] distance at which the goal position is considered to be reached
        self.rotation_tolerance : float = 0.002 # [rad] angle at which the goal rotation is considered to be reached
        self.slowdown_angle : float = 0.25 # [rad] angle at which the slowdown begins. might take longer to reach the desired orientation but will allow for higher precision
        
        self.slowdown_x : float = 3.0 # [m] defines a boxes x-axis that causes slowdowns to the robots speed if objects enter it
        self.slowdown_y : float = 1.4 # [m] defines a boxes y-axis that causes slowdowns to the robots speed if objects enter it
        self.stopping_x: float = 1.50 # [m]defines a box that x-axis causes a stop to the robots speed if objects enter it
        self.stopping_y: float = 0.90 # [m] defines a box that y-axis causes a stop to the robots speed if objects enter it
        self.robot_size_x : float = 1.25 # [m] robot size along x-axis. will igonore laser scans values within this range
        self.robot_size_y : float = 0.85 # [m] robot size along y-axis. will igonore laser scans values within this range
        # ---- End Config ----

        self.stop_moving : bool = False
        
        self.upper_linear_limit : float = self.max_linear_speed
        self.lower_linear_limit : float = -self.max_linear_speed
        self.upper_rotation_limit : float = self.max_angular_speed
        self.lower_rotation_limit : float = -self.max_angular_speed

        self.robot_yaw : float = 0.0
        self.k_linear : float = 1.0 # higher value -> faster linear movement value > 1.0 might lead to start/stop behaviour
        self.k_angular : float = 2.0 # higher value -> faster turnings
        
        self.scanner : LaserScanner = LaserScanner(self.robot_id)
        self.status_publisher = rospy.Publisher('/pmadmu_planner/follower_status', FollowerFeedback, queue_size=10, latch=True)
        rate : rospy.Rate = rospy.Rate(100)
        while not self.scanner.initialized:
            rate.sleep()
        rospy.Subscriber('pmadmu_planner/trajectories', Trajectories, self.trajectory_update)
        rospy.Subscriber(f'/mir{self.robot_id}/robot_pose', Pose, self.update_pose)
        rospy.Subscriber(f'/mir{self.robot_id}/mir_pose_simple', Pose, self.update_pose)
        rospy.Subscriber(f'/mir{self.robot_id}/scan', LaserScan, self.safety_limit_update)
        rospy.Subscriber('/pmadmu_planner/follower_status', FollowerFeedback, self.receive_feedback)
        rospy.Subscriber('/pmadmu_planner/static_obstacles', OccupancyGrid, self.update_costmap)

        self.goal_publisher = rospy.Publisher(f'/mir{self.robot_id}/move_base_simple/goal', PoseStamped, queue_size=10)
        self.cmd_publisher = rospy.Publisher(f'/mir{self.robot_id}/cmd_vel', Twist, queue_size=10)
        

        self.reached_waypoints : int = 0
        self.target_waypoint : Waypoint | None = None
        self.record = False


        self.csv_file = self.initialize_csv()
        self.start_time = time.time()
        rospy.Timer(rospy.Duration.from_sec(0.1), self.log_position, oneshot=False)

        rospy.Timer(rospy.Duration(1, 0), self.position_watchdog, oneshot=False)
        
        self.stop_robot() #for safety reasons, don't remove this! otherwise the robot may start follwing an old deprecated plan
        rospy.spin()
        self.csv_file.close()
        return None
    

    def initialize_csv(self):
        csv_path = os.path.join(os.path.expanduser('~'), f'robot{self.robot_id}_positions.csv')
        file_exists = os.path.isfile(csv_path)
        csv_file = open(csv_path, mode='a', newline='')
        self.csv_writer = csv.writer(csv_file)
        if not file_exists:
            self.csv_writer.writerow(['time', 'x', 'y'])
        return csv_file
    

    def log_position(self, _) -> None:
        if self.robot_pose is None:
            return None
        if self.trajectory is None:
            return None
        if self.record == False:
            self.start_time = time.time()
            return None            
        current_time : float = time.time() - self.start_time
        current_x : float = self.robot_pose.position.x
        current_y : float = self.robot_pose.position.y
        self.csv_writer.writerow([current_time, current_x, current_y])
        return None
    

    def receive_feedback(self, feedback : FollowerFeedback) -> None:
        if feedback.robot_id == self.robot_id:
            return None
        if feedback.status in [feedback.LOST_WAYPOINT, feedback.OUTSIDE_RESERVED_AREA, feedback.PATH_BLOCKED, feedback.PLANNING_FAILED]:
            self.stop_moving = True
            self.stop_robot()
        return None
    

    def update_costmap(self, costmap: OccupancyGrid) -> None:
        self.costmap = costmap
        return None
    

    def position_watchdog(self, _) -> None:
        if self.robot_pose is None:
            return None
        if self.trajectory is None:
            return None
        if self.trajectory.occupied_positions is None:
            return None
        if self.stop_moving:
            return None
        
        transform_world_to_pixel = rospy.ServiceProxy('/pmadmu_planner/world_to_pixel', TransformWorldToPixel)
        w2p_response : TransformWorldToPixelResponse = transform_world_to_pixel([self.robot_pose.position.x], [self.robot_pose.position.y])
        if len(w2p_response.x_pixel) == 0 or len(w2p_response.y_pixel) == 0:
            rospy.logwarn(f"[Follower {self.robot_id}] position watchdog failed to convert position to pixel space.")
            return None
        x_pixel : int = int(w2p_response.x_pixel[0])
        y_pixel : int = int(w2p_response.y_pixel[0])

        # Check if both positon and time are valid (robot must be inside the currently reserved area)
        current_waypoint : Waypoint | None = None
        is_in_reserved_area : bool = False
        for wp in self.trajectory.occupied_positions:
            waypoint : Waypoint = wp
            is_position_valid : bool = ((waypoint.pixel_position.x == x_pixel) and (waypoint.pixel_position.y == y_pixel))
            if is_position_valid:
                current_waypoint = wp
                #rospy.loginfo(f"[Follower {self.robot_id}] occupied from {waypoint.occupied_from} < {time.time() - self.trajectory_start_time} < {waypoint.occupied_until}")
                is_in_reserved_area = waypoint.occupied_from <= time.time() - self.trajectory_start_time <= waypoint.occupied_until
                break

        if not is_in_reserved_area:
            rospy.logerr(f"[Follower {self.robot_id}] is outside the reserved area! Stopping and requesting a new plan!")
            self.stop_robot()
            feedback : FollowerFeedback = FollowerFeedback()
            feedback.robot_id = self.robot_id
            feedback.status = feedback.OUTSIDE_RESERVED_AREA
            self.stop_moving = True
            self.status_publisher.publish(feedback)

        # Check if there are obstacles in the way -> replan if a collision is expected within a given timeframe
        if self.costmap is not None and current_waypoint is not None:
            costmap_data : np.ndarray = np.reshape(self.costmap.data, (self.costmap.info.height, self.costmap.info.width))
            for wp in self.trajectory.occupied_positions:
                waypoint : Waypoint = wp
                if current_waypoint.occupied_from <= waypoint.occupied_from <= current_waypoint.occupied_from + self.lookahead_time:
                    if costmap_data[waypoint.pixel_position.x, waypoint.pixel_position.y] == 0:
                        rospy.logwarn(f"[Follower {self.robot_id}] Object is blocking the path, Collision in {waypoint.occupied_from - current_waypoint.occupied_from}s (<{self.lookahead_time}s inbound!")
                        self.stop_moving = True
                        feedback : FollowerFeedback = FollowerFeedback()
                        feedback.robot_id = self.robot_id
                        feedback.status = feedback.PATH_BLOCKED
                        self.stop_robot()
                        self.status_publisher.publish(feedback)
                        return None
        return None


    def safety_limit_update(self, _ :LaserScan) -> None:
        _upper_linear_limit : float = self.max_linear_speed
        _lower_linear_limit : float = -self.max_linear_speed
        _upper_rotation_limit : float = self.max_angular_speed
        _lower_rotation_limit : float = -self.max_angular_speed

        laser_points : list[list[float]] = self.scanner.get_laser_points()
        crit_radius : float = np.hypot(self.robot_size_x / 2, self.robot_size_y / 2) * 1.05 # check rotational collision when object gets this close. multiplication factor for safety margin
        for point in laser_points:
            #* INVALID DATA FILTERING
            # Ignore Points that are inside our robot. this may happen if the laser scanner recognizes robot parts as obstacle
            if self.robot_size_x / 2 > abs(point[0]) and self.robot_size_y / 2 > abs(point[1]):
                continue
            #* LINEAR COLLISION AVOIDANCE
            # Check for very close obstacles -> stop
            if abs(point[0]) < self.stopping_x / 2 and abs(point[1]) < self.robot_size_y / 2:
                if point[0] > 0:
                    _upper_linear_limit = 0.0
                else:
                    _lower_linear_limit = 0.0
            # Check for somewhat close obstacles -> slowdown
            elif abs(point[0]) < self.slowdown_x / 2 and abs(point[1]) < self.slowdown_y / 2:
                slope: float = self.max_linear_speed / ((self.slowdown_x - self.stopping_x) / 2)
                offset : float = -slope * self.stopping_x / 2

                x_vel : float = (slope * abs(point[0]) + offset)
                x_vel = min(x_vel, self.max_linear_speed)
                x_vel = max(x_vel, 0.05 * self.max_linear_speed) # define min velocity
                
                if point[0] > 0:
                    _upper_linear_limit = min(_upper_linear_limit, x_vel)
                else:
                    _lower_linear_limit = max(_lower_linear_limit, -x_vel)

            #* ROTATIONAL DATA FILTERING
            dist_from_robot: float = np.hypot(point[0], point[1])
            if dist_from_robot < crit_radius:
                #* CHECK FOR TOTAL STOP
                # check if object is in critical distance and we need to block all rotations that may lead to a collision
                is_clockwise_collision: bool = (np.sign(point[0]) == np.sign(point[1]))
                if abs(point[1]) < self.stopping_y / 2 and point[0] < self.stopping_x / 2:
                    # critical zone in the middle of the robot where both rotation directions may lead to a collsion
                    if abs(point[0]) < 0.1 * self.robot_size_x:
                        _upper_rotation_limit = 0
                        _lower_rotation_limit = 0
                    ## collisions near the robots corner only block one rotational direction
                    if is_clockwise_collision:
                        _upper_rotation_limit = 0
                    else:
                        _lower_rotation_limit = 0
                #* CHECK FOR SLOWDOWN
                dist_to_robot_flank : float = max(0, abs(point[1]) - self.robot_size_y / 2)
                angle : float = np.arctan2(dist_to_robot_flank, point[0])
                
                #angle : float = np.arctan2(point[1] - self.robot_size_y * np.sign(point[1]), point[0])
                speed_factor : float = 1.0 - 2 * abs(0.5 - angle/np.pi)
                if is_clockwise_collision:
                    _upper_rotation_limit = min(_upper_rotation_limit, speed_factor * self.max_angular_speed)
                else:
                    _lower_rotation_limit = max(_lower_rotation_limit, -speed_factor * self.max_angular_speed)
          
        self.upper_linear_limit = _upper_linear_limit
        self.lower_linear_limit = _lower_linear_limit
        self.upper_rotation_limit = _upper_rotation_limit
        self.lower_rotation_limit = _lower_rotation_limit  
        return None


    def stop_robot(self) -> None:
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.linear.y = 0.0
        twist_msg.linear.z = 0.0
        twist_msg.angular.x = 0.0
        twist_msg.angular.y = 0.0
        twist_msg.angular.z = 0.0
        self.cmd_publisher.publish(twist_msg)
        self.stop_moving = True
        return None
    

    def update_pose(self, pose: Pose) -> None:
        self.robot_pose = pose
        (_, _, self.robot_yaw) = euler_from_quaternion([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        return None


    def get_distance(self, robot_pose : Pose, waypoint : Waypoint) -> float:
        return np.hypot(robot_pose.position.x - waypoint.world_position.position.x, robot_pose.position.y - waypoint.world_position.position.y)


    def get_angle(self, robot_pose : Pose, waypoint : Waypoint) -> float:
        return np.arctan2(waypoint.world_position.position.y - robot_pose.position.y, waypoint.world_position.position.x - robot_pose.position.x)


    def update_target_point(self) -> Waypoint | None:
        if self.trajectory is None or self.robot_pose is None or self.trajectory.path is None:
            rospy.logwarn(f"[Follower {self.robot_id}] Can't update the target point since Planner is not initialized")
            return None
        
        if self.trajectory.planner_id == 1 and self.target_waypoint is not None:
        
            marker_pub = rospy.Publisher('/pmadmu_planner/pure_pursuit', Marker, queue_size=10, latch=True)
            marker: Marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "pmadmu_planner"
            marker.id = self.trajectory.planner_id
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            #marker.lifetime = rospy.Duration(0, int(1_000_000_000 / time_factor))
            marker.lifetime = rospy.Duration(10, 0)
            marker.pose.orientation.w = 1.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.position = self.target_waypoint.world_position.position
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            point : Point = Point()
            point.x = self.target_waypoint.world_position.position.x
            point.y = self.target_waypoint.world_position.position.y
            point.z = self.target_waypoint.world_position.position.z
            marker.points = [point]
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker_pub.publish(marker)

        if self.target_waypoint is None:
            self.target_waypoint = self.trajectory.start_waypoint
        if self.target_waypoint == self.trajectory.goal_waypoint:
            return self.trajectory.goal_waypoint
        
        distance_to_target : float = self.get_distance(self.robot_pose, self.target_waypoint)
        costmap_data : np.ndarray | None = None
        if self.costmap is not None:
            costmap_data = np.reshape(self.costmap.data, (self.costmap.info.height, self.costmap.info.width))
        while (distance_to_target < self.lookahead_distance) and len(self.trajectory.path) > self.reached_waypoints:
            # Check for valid Occupation Time
            waypoint_limit = self.reached_waypoints + 1
            if waypoint_limit >= len(self.trajectory.path):
                waypoint_limit = len(self.trajectory.path) - 1 
            if time.time() - self.trajectory_start_time <= self.trajectory.path[waypoint_limit].occupied_from:
                #rospy.loginfo(f"[Follower {self.robot_id}] is too fast. expected to arrive at target at {time.time() + (distance_to_target / self.max_linear_speed)} but waypoint is occupied from {self.target_waypoint.occupied_from + self.trajectory_start_time}")
                break

            if costmap_data is not None and costmap_data[self.target_waypoint.pixel_position.x, self.target_waypoint.pixel_position.y] == 0:
                rospy.loginfo(f"[Follower {self.robot_id}] Waypoint is inside an obstacle")
                break
            self.reached_waypoints += 1
            
            if len(self.trajectory.path) <= self.reached_waypoints:
                continue
            self.target_waypoint = self.trajectory.path[self.reached_waypoints]
            distance_to_target : float = self.get_distance(self.robot_pose, self.target_waypoint)
        if self.reached_waypoints < len(self.trajectory.path):
            return self.trajectory.path[self.reached_waypoints]
        return None


    def trajectory_update(self, trajectories: Trajectories) -> None:
        if trajectories.trajectories is None:
            return None
        if self.robot_pose is None:
            rospy.logwarn(f"[Follower {self.robot_id}] Waiting for Robot Pose")
            return None
        self.trajectory_start_time = 0.0
        self.trajectory : Trajectory | None = None
        for trajectory in trajectories.trajectories:
            tray : Trajectory = trajectory
            if tray.planner_id == self.robot_id:
                self.trajectory = tray
                self.trajectory_start_time = trajectories.timestamp
                break

        if self.trajectory is None:
            rospy.loginfo(f"[Follower {self.robot_id}] No new Trajectory.")
            return None
        rospy.loginfo(f"[Follower {self.robot_id}] Received a new Trajectory")
        self.record = True
        
        self.stop_moving = False
        self.reached_waypoints = 0
        self.target_waypoint = trajectory.start_waypoint
        rate : rospy.Rate = rospy.Rate(100)
        
        feedback : FollowerFeedback = FollowerFeedback()
        feedback.robot_id = self.robot_id
        feedback.status = feedback.FOLLOWING
        self.status_publisher.publish(feedback)
        
        distance_to_goal : float = self.get_distance(self.robot_pose, self.trajectory.goal_waypoint)
        if (distance_to_goal > self.goal_tolerance * 3.0):
            while not rospy.is_shutdown():
                if self.follow_trajectory():
                    break
                if self.stop_moving:
                    return None
                rate.sleep()
        else:
            rospy.loginfo(f"[Follower {self.robot_id}] Received a new Trajectory but is already close enough to the goal position ({distance_to_goal:3f}m)")


        while not rospy.is_shutdown():
            if self.rotate(self.trajectory.goal_waypoint.world_position.orientation.z):
                break
            if self.stop_moving:
                return None
            rate.sleep()
        
        rospy.loginfo(f"[Follower {self.robot_id}] Done!")
        self.record = False
        self.stop_robot()
        return None
    

    def control_speeds(self, distance_to_target, steering_angle) -> tuple[float, float]:
        linear_speed = min(self.max_linear_speed, self.k_linear * distance_to_target)
        angular_speed = self.k_angular * steering_angle
        return linear_speed, angular_speed
    

    def get_min_angle(self, angle_1 : float, angle_2 : float) -> float:
        min_angle : float = angle_1 - angle_2
        min_angle -= 2*np.pi if min_angle > np.pi else 0.0
        min_angle += 2*np.pi if min_angle < -np.pi else 0.0
        return min_angle
    

    def rotate(self, target_rotation : float) -> bool:
        angle_error : float = self.get_min_angle(target_rotation, self.robot_yaw)
            
        #rospy.loginfo(f"angle error {angle_error}")

        rotation_direction : float = np.sign(angle_error)
        twist_msg = Twist()
        twist_msg.linear.x = 0.0

        if abs(angle_error) < self.rotation_tolerance:
            twist_msg.angular.z = 0.0
            self.cmd_publisher.publish(twist_msg)
            rospy.loginfo(f"[Follower {self.robot_id}] Reached the Goal Orientation with a tolerance of {abs(angle_error):.5f} rad")
            feedback : FollowerFeedback = FollowerFeedback()
            feedback.robot_id = self.robot_id
            feedback.status = feedback.DONE
            self.status_publisher.publish(feedback)
            return True
        
        angular_speed : float = self.max_angular_speed
        min_angular_speed : float = 0.01 * self.max_angular_speed # min angular speed = 1% of max speed. #todo: make this a parameter (?)
        if abs(angle_error) < self.slowdown_angle:
            angular_speed = self.max_angular_speed * min((abs(angle_error) / self.slowdown_angle), 1.0)

        angular_speed = min(self.max_angular_speed, max(min_angular_speed, angular_speed))
        angular_speed *= rotation_direction

        angular_speed = max(angular_speed, self.lower_rotation_limit)
        angular_speed = min(angular_speed, self.upper_rotation_limit)

        twist_msg.angular.z = angular_speed
        self.cmd_publisher.publish(twist_msg)
        return False



    def follow_trajectory(self) -> bool:
        if self.robot_pose is None or self.target_waypoint is None:
            #rospy.logwarn(f"[Follower {self.robot_id}] Waiting for initialization")
            return False
        
        target_waypoint : Waypoint | None = self.update_target_point()
        if target_waypoint is None:
            return False
        if self.trajectory is None:
            return False
        
        distance_to_target : float = self.get_distance(self.robot_pose, target_waypoint)
        
        if distance_to_target > 2 * self.lookahead_distance:
            rospy.logerr(f"[Follower {self.robot_id}] got too far away from current waypoint. Stopping and requesting a new plan!")
            self.stop_robot()
            feedback : FollowerFeedback = FollowerFeedback()
            feedback.robot_id = self.robot_id
            feedback.status = feedback.LOST_WAYPOINT
            self.stop_moving = True
            self.status_publisher.publish(feedback)
            return False
        
        if distance_to_target < self.goal_tolerance and self.target_waypoint == self.trajectory.goal_waypoint:
            rospy.loginfo(f"[Follower {self.robot_id}] Reached the Goal Position with a tolerance of {distance_to_target:.3f} m")
            twist_msg = Twist()
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0
            self.cmd_publisher.publish(twist_msg)
            return True

        
        angle_to_target : float = self.get_angle(self.robot_pose, self.target_waypoint)
        if self.target_waypoint == self.trajectory.start_waypoint and not self.target_waypoint == self.trajectory.goal_waypoint and self.trajectory.path is not None and len(self.trajectory.path) > 1:
            angle_to_target = self.get_angle(self.robot_pose, self.trajectory.path[1])
        steering_angle : float = angle_to_target - self.robot_yaw
        if steering_angle > np.pi:
            steering_angle -= 2 * np.pi
        elif steering_angle < -np.pi:
            steering_angle += 2 * np.pi

        distance_to_target = self.get_distance(self.robot_pose, self.target_waypoint)
        linear_speed, angular_speed = self.control_speeds(distance_to_target, steering_angle)
        linear_speed = linear_speed * max(np.cos(2*steering_angle), 0)
        if abs(2*steering_angle) > np.pi / 4:
            linear_speed = 0 # prevents robot from driving backwards at big angles where cos becomes positive again

        # Apply Robot Harware Limits
        linear_speed = min(self.max_linear_speed, max(-self.max_linear_speed, linear_speed))
        angular_speed = min(self.max_angular_speed, max(-self.max_angular_speed, angular_speed))

        # Apply Safety Limits
        linear_speed = max(linear_speed, self.lower_linear_limit)
        linear_speed = min(linear_speed, self.upper_linear_limit)
        angular_speed = min(angular_speed, self.upper_rotation_limit)
        angular_speed = max(angular_speed, self.lower_rotation_limit)

        # Stop Robot from Rotating at a Waiting Position
        if distance_to_target < self.goal_tolerance:
            angular_speed = 0
            linear_speed = 0

        twist_msg = Twist()
        twist_msg.linear.x = linear_speed
        twist_msg.angular.z = angular_speed
        self.cmd_publisher.publish(twist_msg)
        return False

            
if __name__ == '__main__':
    rospy.init_node("path_follower")
    planner_id : int = rospy.get_param('~robot_id', default=-1) # type: ignore
    path_follower : PathFollower = PathFollower(planner_id)