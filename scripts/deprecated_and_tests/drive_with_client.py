#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import actionlib
from typing import List, Tuple
from geometry_msgs.msg import Twist, PoseStamped, Point, Quaternion
import mbf_msgs.msg as mbf_msgs


def move_to_pos(x, y) -> bool:
    goal  : mbf_msgs.MoveBaseGoal = mbf_msgs.MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y
    goal.target_pose.pose.orientation.w = 1.0

    client.send_goal(goal)
    client.wait_for_result()

    if client.get_state() == actionlib.GoalStatus.SUCCEEDED:
        rospy.loginfo("Reached Goal")
        return True
    else:
        rospy.loginfo("Failed to reach goal...")
        return False





if __name__== '__main__':
    rospy.loginfo("INITIALIZING DRIVE NODE")
    rospy.init_node('ma_nav')
    rate = rospy.Rate(10) # Hz

    rospy.loginfo("Setting up Action Client")
    client = actionlib.SimpleActionClient("/mir1/move_base_flex/move_base", mbf_msgs.MoveBaseAction)
    rospy.loginfo("Waiting for Server....")
    client.wait_for_server()
    rospy.loginfo("Connected.")
    
    positions: List[Tuple[float, float]] = [(28, 16), (34, 15), (35, 21), (28, 23)]

    while not rospy.is_shutdown():
        for pos in positions:
            rate.sleep()
            rospy.loginfo("i am running")
            rospy.loginfo(f"driving to {pos[0]}/{pos[1]}")
            goal_reached : bool = move_to_pos(pos[0], pos[1])
            if goal_reached:
                rospy.loginfo(f"reached goal {pos}")
    rospy.loginfo("DRIVE NODE DONE!")
