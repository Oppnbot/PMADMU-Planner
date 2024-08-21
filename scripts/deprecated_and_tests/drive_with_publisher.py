#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import actionlib
from typing import List, Tuple
from geometry_msgs.msg import Twist, PoseStamped, Point, Quaternion
import mbf_msgs.msg as mbf_msgs



def move_to_pos(x, y) -> bool:
    publisher = rospy.Publisher('/mir1/move_base_simple/goal', PoseStamped, queue_size=10)
    pose = PoseStamped()
    pose.header.stamp = rospy.Time.now()
    pose.header.frame_id = 'map'
    pose.pose.position = Point(x, y, 0.0)
    pose.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
    publisher.publish(pose)
    return False




if __name__== '__main__':
    rospy.loginfo("INITIALIZING DRIVE NODE")
    rospy.init_node('ma_nav')
    rate = rospy.Rate(0.1) # Hz
    
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
