#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

from __future__ import annotations

import time
import rospy
import actionlib
import mbf_msgs.msg as mbf_msgs
import numpy as np

from geometry_msgs.msg import Twist, PoseStamped, Point, Quaternion, Pose
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from formation_builder.msg import Trajectory, Waypoint
from nav_msgs.msg import Path

#https://answers.ros.org/question/295495/how-to-follow-a-nav_msgspath/

class Test:
    def __init__(self) -> None:
        rospy.init_node('point_sender', anonymous=True)
        self.robot_pose : Pose
        rospy.Subscriber('/mir1/mir_pose_simple', Pose, self.update_pose)
        #self.path_pub = rospy.Publisher('/mir1/move_base_flex/DWAPlannerROS/global_plan', Path, queue_size=10, latch=True)
        self.path_pub = rospy.Publisher("/mir1/move_base_flex/GlobalPlanner/plan", Path, queue_size=10, latch=True)
        self.goal_pub = rospy.Publisher('/mir1/move_base_flex/current_goal', PoseStamped, queue_size=10, latch=True)
        self.rate = rospy.Rate(1)
        self.rate.sleep()
        
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.logwarn("lets go")
        self.send_points_to_planner()
        return None


    def update_pose(self, pose: Pose) -> None:
        self.robot_pose = pose
        return None
    
    def send_points_to_planner(self):
        
        rospy.logwarn("Sending")
        
        
        path : Path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = rospy.Time.now()

        path.poses = []

        point0 = PoseStamped()
        point0.header.frame_id = 'map'
        point0.pose = self.robot_pose

        point1 = PoseStamped()
        point1.header.frame_id = 'map'
        point1.header.stamp.secs = 1
        point1.pose.position.x = 33.0
        point1.pose.position.y = 22.0
        point1.pose.position.z = 0.0
        point1.pose.orientation.w = 1.0

        point2 = PoseStamped()
        point2.header.frame_id = 'map'
        point1.header.stamp.secs = 2
        point2.pose.position.x = 25.0
        point2.pose.position.y = 22.0
        point2.pose.position.z = 0.0
        point2.pose.orientation.w = 1.0

        point3 = PoseStamped()
        point3.header.frame_id = 'map'
        point1.header.stamp.secs = 3
        point3.pose.position.x = 25.0
        point3.pose.position.y = 25.0
        point3.pose.position.z = 0.0
        point3.pose.orientation.w = 1.0

        path.poses = [point0, point1, point2, point3]
        
        self.goal_pub.publish(point3)
        quick_rate = rospy.Rate(100)
        self.path_pub.publish(path)

        while not rospy.is_shutdown():
            self.path_pub.publish(path)

            
            quick_rate.sleep()
        
        self.rate.sleep()
        return None



if __name__ == '__main__':

    test = Test()


    