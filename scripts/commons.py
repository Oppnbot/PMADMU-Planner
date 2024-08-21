#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import rospy


class TrajectoryData():
    def __init__(self, planner_id : int, waypoints : list[Waypoint]):
        """
        The Trajectory describes a robots path while also including informations about timings. Each Path consists of multiple waypoints
        :params planner_id: this correlates the path to one of the robots. Can also be used to determine the paths color in the visualization.
        :params waypoints: the grid cells that make up the robots paths. they also include the timings of the robot entering/leaving the cell.
        """
        self.planner_id : int = planner_id
        self.waypoints : list[Waypoint] = waypoints

        self.start : Waypoint | None = waypoints[0] if waypoints else None
        self.goal : Waypoint | None = waypoints[-1] if waypoints else None

        if not waypoints:
            rospy.logwarn(f"Planner {planner_id} tried to generate a trajectory with 0 waypoints.")


class Waypoint():
    def __init__(self, pixel_pos : tuple[int, int], occupied_from: float, occupied_until : float = float('inf'), world_pos : tuple[float, float]|None = None, previous_waypoint:Waypoint|None = None):
        """
        Each grid cell the robot passes counts as a Waypoint. Multiple Waypoints make up the robots path.
        :params world_pos: the (x, y) position in world coordinates [m]; used to navigate the robot
        :params pixel_pos: the (x, y) position in pixel coordinates [px]; used to find a path
        :params occupied_from: time when waypoint first becomes occupied, making it unavailable for othe
        :params occupied_until: time when waypoint becomes free, making it available for other robots [s]
        """
        self.world_pos : tuple[float, float] | None = world_pos # the (x, y) position in world coordinates [m]; used to navigate the robot
        self.pixel_pos : tuple[int, int] = pixel_pos            # the (x, y) position in pixel coordinates [px]; used to find a path
        self.occupied_from: float = occupied_from               # time when waypoint first becomes occupied, making it unavailable for other robots [s]
        self.occupied_until: float = occupied_until              # time when waypoint becomes free, making it available for other robots [s]
        self.previous_waypoint : Waypoint|None = previous_waypoint

    def __eq__(self, __value: Waypoint) -> bool:
        return self.pixel_pos == __value.pixel_pos

    def __lt__(self, other : Waypoint):
        return self.occupied_from < other.occupied_from


