from collections import defaultdict
from tqdm import tqdm
import random
import shapely.geometry
# import cupy as np
import numpy as np



class InformedRRTStarCpp:
    """imports the class from Informed_RRT.cc"""




class InformedRRTStar:
    def __init__(self, vectormap, min_x, max_x, min_y, max_y):
        self.start = None
        self.goal = None
        self.goal_radius = 0.5
        self.path = []
        # cupy array of 2d points
        self.vertices = []
        # parents vector
        self.parents = []
        # costs vector
        self.costs = []
        # costs distance
        self.costs_distance = []
        # path vector
        self.path = []
        self.goal_index = -1
        self.vectormap = np.array([[[line['p0']['x'], line['p0']['y']], [line['p1']['x'], line['p1']['y']]] for line in vectormap])
        self.num_iterations = 5000
        self.max_iterations = 100000
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.radius = 1
        self.step_size = 0.5
        self.safety_margin = 0.2
        
        
        

    

    def SampleRandomPoint(self, radius, center):
        min_x = max(center[0] - radius, self.min_x)
        max_x = min(center[0] + radius, self.max_x)
        min_y = max(center[1] - radius, self.min_y)
        max_y = min(center[1] + radius, self.max_y)
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        return np.array([x, y])
    
    def FindNearestVertex(self, point):
        if len(self.vertices) == 0:
            return None
        return np.argmin(np.linalg.norm(self.vertices - point, axis=1))
    

    def Steer(self, index, point):
        direction = (point - self.vertices[index])
        direction = self.step_size * direction / np.linalg.norm(direction)
        return self.vertices[index] + direction
    
    def ClosestDistanceToWall(self, start, end):
        line = shapely.geometry.LineString([start, end])
        min_distance = float('inf')
        for wall in self.vectormap:
            # wall has shape (2, 2)
            wall = shapely.geometry.LineString(wall)
            distance = line.distance(wall)
            if distance < min_distance:
                min_distance = distance
        return min_distance

    def IsCollision(self, closest_distance):
        return closest_distance < self.safety_margin
    
    def IsValidVertex(self, vertex):
        closest_distance = self.ClosestDistanceToWall(vertex, vertex)
        return not self.IsCollision(closest_distance)
    

    def Cost(self, start, end, closest_distance):
        return np.linalg.norm(end - start) + 5/closest_distance 
    
    def FindNearestVertexInRadius(self, point, vertices_in_radius):
        if len(vertices_in_radius) == 0:
            return None
        min_cost = float('inf')
        nearest_vertex = None
        for vertex_index in vertices_in_radius:
            vertex = self.vertices[vertex_index]
            closest_distance = self.ClosestDistanceToWall(point, vertex)
            if not self.IsCollision(closest_distance):
                cost = self.Cost(point, vertex, closest_distance)
                if cost < min_cost:
                    min_cost = cost
                    nearest_vertex = vertex_index
        return nearest_vertex
    
    def FindVerticesInRadius(self, point, radius):
        return np.where(np.linalg.norm(self.vertices - point, axis=1) < radius)[0]


    def SampleUnitBall(self):
        while True:
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            if x ** 2 + y ** 2 <= 1:
                return np.array([x, y, 0.0])

    def RotationToWorldFrame(self, x_start, x_goal, L):
            a1 = np.array([[(x_goal[0] - x_start[0]) / L],
                        [(x_goal[1] - x_start[1]) / L], [0.0]])
            e1 = np.array([[1.0], [0.0], [0.0]])
            M = a1 @ e1.T
            U, _, V_T = np.linalg.svd(M, True, True)
            C = U @ np.diag([1.0, 1.0, np.linalg.det(U) * np.linalg.det(V_T.T)]) @ V_T
            return C

    # informed rrt sampling which samples points within an ellipse which encompasses the start and goal point and the current optimal solution
    def InformedRRTSampling(self, c_min, rotation_to_world):
        start_point = self.vertices[0]
        goal_point = self.vertices[self.goal_index]
        c_max = self.costs_distance[self.goal_index]
        center = (start_point + goal_point) / 2
        center = np.array([center[0], center[1], 0.0])
        r_1 = c_max / 2
        r_2 = (c_max ** 2 - c_min ** 2) / 2
        r_3 = (c_max ** 2 - c_min ** 2) / 2
        r = np.array([r_1, r_2, r_3])
        L = np.diag(r)
        C = rotation_to_world
        x_ball = self.SampleUnitBall()
        x_rand = np.dot(np.dot(C, L), x_ball) + center
        return [x_rand[0], x_rand[1]]

    def Clear(self):
        self.vertices, self.parents, self.costs, self.costs_distance, self.path = [], [], [], [], []

    def Rewire(self, new_vertex, radius_indices):
        new_vertex_index = len(self.vertices) - 1
        new_vertex_cost = self.costs[new_vertex_index]
        new_vertex_cost_distance = self.costs_distance[new_vertex_index]
        for vertex_near in radius_indices:
            closest_distance_near_new = self.ClosestDistanceToWall(self.vertices[vertex_near], new_vertex)
            if self.IsCollision(closest_distance_near_new):
                cost = new_vertex_cost + self.Cost(new_vertex, self.vertices[vertex_near], closest_distance_near_new)
                cost_distance = new_vertex_cost_distance + np.linalg.norm(new_vertex - self.vertices[vertex_near])
                if cost < self.costs[vertex_near]:
                    self.parents[vertex_near] = new_vertex_index
                    self.costs[vertex_near] = cost
                    self.costs_distance[vertex_near] = cost_distance
    
    def SetStart(self, start):
        self.start = start
        self.vertices.append(start)
        self.parents.append(0)
        self.costs.append(0)
        self.costs_distance.append(0)

    def SetGoal(self, goal):
        self.goal = goal
        self.goal_index = -1

    def SmoothPath(self, path):
        if len(path) < 2:
            return path
        smoothed_path = []
        # connect the farthest vertices with no collision
        start_position = 0
        while start_position < len(path):
            start_vertex = path[start_position]
            smoothed_path.append(start_vertex)
            for end_position in range(len(path) - 1, start_position, -1):
                end_vertex = path[end_position]
                closest_distance = self.ClosestDistanceToWall(start_vertex, end_vertex)
                cost_start_end = self.Cost(start_vertex, end_vertex, closest_distance)
                if not self.IsCollision(closest_distance) and cost_start_end + self.costs[start_position] < self.costs[end_position]:
                    smoothed_path.append(end_vertex)
                    # update the costs of future vertices
                    cost_reduction = self.costs[end_position] - (self.costs[start_position] + cost_start_end)
                    np.subtract(self.costs[end_position:], cost_reduction, out = self.costs[end_position:])
                    start_position = end_position
                    break
            start_position += 1
        return smoothed_path
            

    def SetPath(self):
        if self.goal_index == -1:
            return
        path_init = [self.vertices[self.goal_index]]
        path_indices = [self.goal_index]
        curr_index = self.goal_index
        # backtrace the path
        while curr_index != 0:
            parent_index, curr_index = self.parents[curr_index], parent_index
            path_init.append(self.vertices[curr_index])
            path_indices.append(curr_index)
        
        path_init.reverse()
        path_indices.reverse()

        # add the goal as the last point
        self.vertices.append(self.goal)
        self.parents.append(self.goal_index)
        goal_closest_distance = self.ClosestDistanceToWall(self.vertices[self.goal_index], self.goal)
        self.costs.append(self.Cost(self.vertices[self.goal_index], self.goal, goal_closest_distance) + self.costs[self.goal_index])
        self.costs_distance.append(self.costs_distance[self.goal_index] + np.linalg.norm(self.goal - self.vertices[self.goal_index]))
        path_init.append(self.goal)
        path_indices.append(len(self.vertices) - 1)

        # smooth the path
        self.path = self.SmoothPath(path_init)

    def GetPath(self):
        return self.path

    def GetTree(self):
        tree = None
        for i in range(len(self.vertices)):
            parent = self.parents[i]
            if parent != i:
                tree.append([self.vertices[i], self.vertices[parent]])
        return tree
                

     
    # implement the informed rrt algorithm
    def Plan(self, start, goal, num_iterations=100):
        self.Clear()
        self.SetStart(start)
        self.SetGoal(goal)
        goal_found  = False
        curr_radius = 0.5
        center = start
        self.step_size = 0.5
        iter = 0
        c_min = np.linalg.norm(start - goal)
        rotation_to_world = self.RotationToWorldFrame(start, goal, c_min)
        iters_after_goal = 0
        progress_bar = tqdm(total=self.max_iterations)
        # run the algorithm for a certain number of iterations
        while iters_after_goal < num_iterations and iter <= self.max_iterations:
            progress_bar.update(1)
            # sample and find the nearest vertex
            random_point = None
            iter += 1
            if goal_found:
                random_point = self.InformedRRTSampling(c_min, rotation_to_world)
                iters_after_goal += 1
            else:
                if iter % 50 == 0:
                    curr_radius *= 1.5
                if iter % 500 == 0:
                    
                    random_point = goal
                    center = start if center == goal else goal
                if iter % 10000 == 0:
                    step_size /= 1.5
                random_point = self.SampleRandomPoint(curr_radius, center)
            
            # make sure random point is not close to any wall
            if not self.IsValidVertex(random_point):
                continue
            # find the nearest vertex
            nearest_vertex_index = self.FindNearestVertex(random_point)
            # steer towards the random point
            new_vertex = self.Steer(nearest_vertex_index, random_point)
            # check if new vertex is valid
            if not self.IsValidVertex(new_vertex):
                continue
            vertices_in_radius = self.FindVerticesInRadius(new_vertex, self.radius)
            nearest_vertex_index = self.FindNearestVertexInRadius(new_vertex, vertices_in_radius)
            # add the new vertex to the vertices
            self.vertices.append(new_vertex)
            self.parents.append(nearest_vertex_index)
            # check if the line segment from nearest to new intersect with the obstacle
            closest_distance = self.ClosestDistanceToWall(self.vertices[nearest_vertex_index], new_vertex)
            self.costs.append(self.Cost(self.vertices[nearest_vertex_index], new_vertex, closest_distance))
            self.costs_distance.append(self.costs_distance[nearest_vertex_index] + np.linalg.norm(new_vertex - self.vertices[nearest_vertex_index]))
            # rewire the tree
            self.Rewire(new_vertex, vertices_in_radius)
            # check if goal is reached
            if (np.linalg.norm(new_vertex - goal) < self.goal_radius):
                if (not goal_found) or (self.costs[-1] < self.costs[self.goal_index]):
                    self.goal_index = len(self.vertices) - 1
                    goal_found = True
        if goal_found:
            self.SetPath()
        return goal_found
    