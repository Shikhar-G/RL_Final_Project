import math
import heapq
import numpy as np
import queue
# Define the Cell class
class Cell:
    def __init__(self):
        self.parent_i = 0  # Parent cell's row index
        self.parent_j = 0  # Parent cell's column index
        self.f = float('inf')  # Total cost of the cell (g + h)
        self.g = float('inf')  # Cost from start to this cell
        self.h = 0  # Heuristic cost from this cell to destination
        


class Astar:
    def __init__(self, grid, vehicle_size):
        self.path = []
        self.ROW = len(grid)
        self.COL = len(grid[0])
        self.margin = 1
        self.vehicle_size = vehicle_size
        self.weighted_grid = self.CreateWeightedGrid(grid)

    # Define the size of the grid
    def findable_area(self, start):
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        visited = dict()
        q = queue.Queue()
        start_l = (start[0], start[1])
        visitedl = [np.asarray[(new_i, new_j)]]
        visited[start_l] = True
        
        q.put(start_l)
        while not q.empty():
            current = q.get()
            for dir in directions:
                new_i = current[0] + dir[0]
                new_j = current[1] + dir[1]
                visitedt = False
                for key in visited.keys():
                    visitedt = visitedt or (key[0] == new_i and key[1] == new_j)
                if self.is_valid(new_i, new_j) and self.is_unblocked(new_i, new_j) and not visitedt:
                    visited[(new_i, new_j)] = True
                    visitedl.append(np.asarray[(new_i, new_j)])
                    q.put((new_i, new_j))
        return visited
    
    def CreateWeightedGrid(self, grid):
        weighted_grid = np.ones((self.ROW, self.COL))
        wall_indices = np.where(grid == 0)
        margin = math.ceil(self.vehicle_size/2 + self.margin)
        for i, j in zip(wall_indices[0], wall_indices[1]):
            weighted_grid[i][j] = float('inf')
            # set all collisions to inf
            for x in range(max(0, i - margin), min(self.ROW, i + margin + 1)):
                for y in range(max(0, j - margin), min(self.COL, j + margin + 1)):
                    weighted_grid[x][y] = float('inf')
            # set costs for the cells around the collision
            distance_out = 10
            for dist in range(margin + 1, margin + 1 + distance_out):
                # set top and bottom row
                x_dist = [-dist, dist]
                for x in x_dist:
                    for y in range(-dist + 1, dist):
                        if self.is_valid(i + x, j + y) and grid[i + x][j + y] == 1:
                            weighted_grid[i + x][j + y] = max(weighted_grid[i + x][j + y], 1 + 5/((dist - margin)**2))
                # check left and right column
                y_dist = [-dist, dist]
                for y in y_dist:
                    for x in range(-dist, dist + 1):
                        if self.is_valid(i + x, j + y) and grid[i + x][j + y] == 1:
                            weighted_grid[i + x][j + y] = max(weighted_grid[i + x][j + y], 1 + 5/((dist - margin)**2))
        return weighted_grid
            
    def IsCollision(self, start, end): 
        # check if the line between start and end is collision free
        # get the cells that the line goes through
        increment = 0.5
        segment_length = np.linalg.norm(end - start)
        d_line_length = round(segment_length/increment)
        theta = math.atan2((end[1] - start[1]),(end[0] - start[0]))
        heading_segment_inc = np.asarray([increment*math.cos(theta), increment*math.sin(theta)])
        for i in range(d_line_length):
            position = np.round(start + i*heading_segment_inc).astype(int)
            if not self.is_valid(position[0], position[1]) or self.weighted_grid[position[0]][position[1]] == float('inf'):
                return True
        return False
        
    def SmoothPath(self):
        path = self.path
        if len(path) < 2:
            return path
        smoothed_path = []
        # connect the farthest vertices with no collision
        start_position = 0
        while start_position < len(path):
            start_vertex = path[start_position]
            smoothed_path.append(start_vertex)
            found_end = False
            for end_position in range(len(path) - 1, start_position, -1):
                end_vertex = path[end_position]
                # closest_distance = self.ClosestDistanceToWall(start_vertex, end_vertex)
                # cost_start_end = self.Cost(start_vertex, end_vertex, closest_distance)
                if not self.IsCollision(start_vertex, end_vertex):
                    # smoothed_path.append(end_vertex)
                    # update the costs of future vertices
                    start_position = end_position
                    found_end = True
                    break
            if not found_end:
                start_position += 1
        return smoothed_path
    
    def get_path(self):
        return self.path
    # Check if a cell is valid (within the grid)
    def is_valid(self, row, col):
        return (row >= 0) and (row < self.ROW) and (col >= 0) and (col < self.COL)
    
    # Check if a cell is unblocked
    def is_unblocked(self, row, col):
        return self.weighted_grid[row][col] != float('inf')


    
    # Check if a cell is the destination
    def is_destination(self, row, col, dest):
        return row == dest[0] and col == dest[1]
    
    # Calculate the heuristic value of a cell (Euclidean distance to destination)
    def calculate_h_value(self, row, col, dest):
        #manhattan distance
        return (abs(row - dest[0]) + abs(col - dest[1]))
        #eudclidean distance
        #return ((row - dest[0]) ** 2 + (col - dest[1]) ** 2) ** 0.5
    
    # Trace the path from source to destination
    def trace_path(self, cell_details, dest):
        print("The Path is ")
        path = []
        row = dest[0]
        col = dest[1]
    
        # Trace the path from destination to source using parent cells
        while not (cell_details[row][col].parent_i == row and cell_details[row][col].parent_j == col):
            path.append(np.asarray([row, col]))
            temp_row = cell_details[row][col].parent_i
            temp_col = cell_details[row][col].parent_j
            row = temp_row
            col = temp_col
    
        # Add the source cell to the path
        path.append(np.asarray([row, col]))
        # Reverse the path to get the path from source to destination
        path.reverse()
    
        # Print the path
        # for i in path:
        #     print("->", i, end=" ")
        # print()'
        self.path = path
    
    # Implement the A* search algorithm
    def a_star_search(self, src, dest):
        # Check if the source and destination are valid
        if not self.is_valid(src[0], src[1]) or not self.is_valid(dest[0], dest[1]):
            print("Source or destination is invalid")
            return False
    
        # Check if the source and destination are unblocked
        if not self.is_unblocked(src[0], src[1]) or not self.is_unblocked(dest[0], dest[1]):
            print("Source or the destination is blocked")
            return False
    
        # Check if we are already at the destination
        if self.is_destination(src[0], src[1], dest):
            print("We are already at the destination")
            return True
    
        # Initialize the closed list (visited cells)
        closed_list = [[False for _ in range(self.COL)] for _ in range(self.ROW)]
        # Initialize the details of each cell
        cell_details = [[Cell() for _ in range(self.COL)] for _ in range(self.ROW)]
    
        # Initialize the start cell details
        i = src[0]
        j = src[1]
        cell_details[i][j].f = 0
        cell_details[i][j].g = 0
        cell_details[i][j].h = 0
        cell_details[i][j].parent_i = i
        cell_details[i][j].parent_j = j
    
        # Initialize the open list (cells to be visited) with the start cell
        open_list = []
        heapq.heappush(open_list, (0.0, i, j))
    
        # Initialize the flag for whether destination is found
        found_dest = False
    
        # Main loop of A* search algorithm
        while len(open_list) > 0:
            # Pop the cell with the smallest f value from the open list
            p = heapq.heappop(open_list)
    
            # Mark the cell as visited
            i = p[1]
            j = p[2]
            closed_list[i][j] = True
    
            # For each direction, check the successors
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
            for dir in directions:
                new_i = i + dir[0]
                new_j = j + dir[1]
    
                # If the successor is valid, unblocked, and not visited
                if self.is_valid(new_i, new_j) and self.is_unblocked(new_i, new_j) and not closed_list[new_i][new_j]:
                    # If the successor is the destination
                    if self.is_destination(new_i, new_j, dest):
                        # Set the parent of the destination cell
                        cell_details[new_i][new_j].parent_i = i
                        cell_details[new_i][new_j].parent_j = j
                        print("The destination cell is found")
                        # Trace and print the path from source to destination
                        self.trace_path(cell_details, dest)
                        found_dest = True
                        return True
                    else:
                        # Calculate the new f, g, and h values
                        g_new = cell_details[i][j].g + self.weighted_grid[new_i][new_j]
                        h_new = self.calculate_h_value(new_i, new_j, dest)
                        f_new = g_new + h_new
    
                        # If the cell is not in the open list or the new f value is smaller
                        if cell_details[new_i][new_j].f == float('inf') or cell_details[new_i][new_j].f > f_new:
                            # Add the cell to the open list
                            heapq.heappush(open_list, (f_new, new_i, new_j))
                            # Update the cell details
                            cell_details[new_i][new_j].f = f_new
                            cell_details[new_i][new_j].g = g_new
                            cell_details[new_i][new_j].h = h_new
                            cell_details[new_i][new_j].parent_i = i
                            cell_details[new_i][new_j].parent_j = j

        # If the destination is not found after visiting all cells
        if not found_dest:
            print("Failed to find the destination cell")
        return False