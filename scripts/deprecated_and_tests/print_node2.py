#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy

def print_message():
    rospy.loginfo("Hello World, i am alive root! :)")

if __name__ == '__main__':
    rospy.init_node('print_node')
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        print_message()
        rate.sleep()