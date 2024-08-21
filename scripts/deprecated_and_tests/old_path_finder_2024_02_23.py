    #!deprecated
    def update_occupation_dict(self, trajectory: TrajectoryData) -> None:
        for waypoint in trajectory.waypoints:
            self.occupation[waypoint.pixel_pos].append((waypoint.occupied_from, waypoint.occupied_until))
        return None


    def path_finder_old(self, static_obstacles: np.ndarray, start_pos: tuple[int, int], goal_pos: tuple[int, int], occupation: dict[tuple[int, int], list[tuple[float, float]]]) -> TrajectoryData:
        rospy.logerr("old deprecated funtion")
        return TrajectoryData(-1, [])
        rospy.loginfo(f"Planner {self.id} Starting wavefront expansion")
        start_time = time.time()

        heap: list[tuple[float, tuple[int, int]]] = [(0, start_pos)]
        rows: int = static_obstacles.shape[0]
        cols: int = static_obstacles.shape[1]

        direct_neighbors: list[tuple[int, int]] = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        diagonal_neighbors: list[tuple[int, int]] = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
        neighbors: list[tuple[int, int]] = direct_neighbors
        if self.allow_diagonals:
            neighbors += diagonal_neighbors

        timings: np.ndarray = np.zeros((rows, cols)) - 1
        timings[start_pos[0], start_pos[1]] = 0.0
        predecessors: dict[tuple[int, int], tuple[int, int]] = {}

        iterations: int = 0

        

        
        #* --- Generate Timings ---
        while heap:
            iterations += 1
            current_cost, current_element = heapq.heappop(heap)

            if self.dynamic_visualization:
                fb_visualizer.draw_timings(timings, static_obstacles, start_pos, goal_pos)

            if iterations % 1000 == 0:
                rospy.loginfo(f"planner {self.id}: {iterations} iterations done!")

            if iterations > 500_000:
                rospy.logwarn(f"planner {self.id}: breaking because algorithm reached max iterations")
                break

            if current_element == goal_pos:
                rospy.loginfo(f"planner {self.id}: Reached the goal after {iterations} iterations")
                break

            #current_cost = timings[current_element[0], current_element[1]]

        
            if self.check_dynamic_obstacles and occupation[current_element]:
                is_occupied : bool = False
                occupation_list : list[tuple[float, float]] = occupation[current_element]
                for occ_from, occ_until in occupation_list:
                    if occ_from -10 <= current_cost <= occ_until: #! this is bad because magic number as to match occ time which may not be static
                        rospy.logwarn(f"robot {self.id} collides at position {current_element} after {current_cost}s while waiting. it is occupied between {occ_from}s -> {occ_until}s ")
                        if current_element in predecessors.keys():
                            heapq.heappush(heap, (occ_until + 0.1, predecessors[current_element]))
                        else:
                            rospy.logerr(f"missing key in predecessors {current_element}")
                        is_occupied = True
                        break
                if is_occupied:
                    continue

            for x_neighbor, y_neighbor in neighbors:
                x, y = current_element[0] + x_neighbor, current_element[1] + y_neighbor
                if 0 <= x < rows and 0 <= y < cols and static_obstacles[x, y] != 0: # check for static obstacles / out of bounds
                    driving_cost = current_cost + (1 if abs(x_neighbor + y_neighbor) == 1 else 1.41422)

                    if self.check_dynamic_obstacles and occupation[(x,y)]: # check for dynamic obstacles
                        is_occupied : bool = False
                        occupation_list : list[tuple[float, float]] = occupation[(x,y)]
                        for occ_from, occ_until in occupation_list:
                            if occ_from <= driving_cost <= occ_until:
                                #rospy.logwarn(f"robots {self.id} path crosses at ({x, y}) from after {driving_cost}. it is occupied between {occ_from} to {occ_until} ")
                                heapq.heappush(heap, (occ_until, (x, y)))
                                is_occupied = True
                                break
                        if is_occupied:
                            continue
                    # cell isnt occupied -> add it to heap
                    if driving_cost < timings[x, y] or timings[x, y] < 0:
                        timings[x, y] = driving_cost
                        predecessors[(x, y)] = current_element
                        heapq.heappush(heap, (driving_cost, (x, y)))

        rospy.loginfo(f"planner {self.id}: stopped after a total of {iterations} iterations")
        end_time = time.time()
        elapsed_time = end_time - start_time
        rospy.loginfo(f"planner {self.id}: planned path in {elapsed_time:.3f}s.")
        

        #* --- Reconstruct Path ---
        waypoints : list[Waypoint] = []

        # Reconstruct path from goal to start using the predecessor of each node
        current_node: tuple[int, int] = goal_pos
        previous_node: tuple[int, int] | None = None
        while current_node != start_pos:
            
            if current_node in predecessors.keys():
                current_node = predecessors[current_node]
                waypoint : Waypoint = Waypoint()
                waypoint.pixel_pos = current_node
                waypoint.occupied_from = timings[current_node[0], current_node[1]]
                if previous_node is None:
                    waypoint.occuped_until = float('inf')
                else:
                    #todo: add some uncertainty compensation here
                    #waypoint.occuped_until = timings[previous_node[0], previous_node[1]] # unmodified
                    waypoint.occuped_until = timings[previous_node[0], previous_node[1]] + 10            # defined snake length
                    #waypoint.occuped_until = (timings[previous_node[0], previous_node[1]] + 1) * 1.1   # snakes get longer over time -> uncertainty grows
                waypoints.append(waypoint)
                previous_node = current_node

                if self.dynamic_visualization:
                    fb_visualizer.draw_timings(timings, static_obstacles, start_pos, goal_pos, waypoints)
                
            else:
                rospy.logerr(f"Cant reconstruct path since {current_node} is not in keys; Path might be unvalid.")
                break
        start_point : Waypoint = Waypoint()
        start_point.occupied_from = 0
        start_point.occuped_until = waypoints[-1].occupied_from # todo: add uncertainty here
        start_point.pixel_pos = start_pos
        waypoints.append(start_point)
        waypoints.reverse()

        trajectory_data : TrajectoryData = TrajectoryData(self.id, waypoints)

        rospy.loginfo(f"planner {self.id}: shortest path consists of {len(waypoints)} nodes with a cost of {timings[goal_pos[0], goal_pos[1]]}")

        # Transform Path from Pixel-Space to World-Space for visualization and path following
        transform_pixel_to_world = rospy.ServiceProxy('pixel_to_world', TransformPixelToWorld)
        for waypoint in trajectory_data.waypoints:
            response : TransformPixelToWorldResponse = transform_pixel_to_world(waypoint.pixel_pos[0], waypoint.pixel_pos[1])
            waypoint.world_pos = (response.x_world, response.y_world)
        
        # Visualization
        fb_visualizer.draw_timings(timings, static_obstacles, start_pos, goal_pos, trajectory_data.waypoints)

        return trajectory_data
    