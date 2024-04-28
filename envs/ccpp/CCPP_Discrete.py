import pygame
from gymnasium.spaces import MultiBinary
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import math
from envs.ccpp.Astar import Astar
import cv2


def get_vectormap(map_file):
    f = open(map_file)
    data = json.load(f)
    return data


def get_image_size(vectormap, scaling=10, padding=0):
    # get the size of the image
    x = []
    y = []
    for line in vectormap:
        for point in line:
            x.append(line[point]["x"])
            y.append(line[point]["y"])
    return (
        math.ceil(max(x) - min(x)) * scaling + padding * 2,
        math.ceil(max(y) - min(y)) * scaling + padding * 2,
    )


def get_image_min_max(vectormap):
    # get the size of the image
    x = []
    y = []
    for line in vectormap:
        for point in line:
            x.append(line[point]["x"])
            y.append(line[point]["y"])
    return min(x), min(y), max(x), max(y)


class CCPP_Discrete(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        render_mode=None,
        map_file="CDL_Ground.vectormap.json",
        agent_dims=np.array([1, 1]),
        agent_loc=np.array([0, 0]),
        agent_dir=np.array([0, 1]),
        agent_max_linear_speed=0.3,
        agent_max_angular_speed=1.9,
        scaling=10,
        coverage_radius=1,
        coverage_required=0.95,
    ):
        # get map properties
        self.scaling = scaling
        self.map_padding = self.scaling
        self.vectormap = get_vectormap(map_file)
        self.coverage_required = coverage_required

        # get the size of the image
        self.image_size_x, self.image_size_y = get_image_size(
            self.vectormap, scaling, self.map_padding
        )

        # get min values for the offset
        self.x_min, self.y_min, self.x_max, self.y_max = get_image_min_max(
            self.vectormap
        )

        # initialize map channel
        self.set_map_channel()

        # initialize coverage channel
        self.coverage_channel = np.zeros(
            (self.image_size_x, self.image_size_y), dtype=np.uint8
        )

        # initialize agent channel
        self.agent_dims = np.array(
            [math.ceil(agent_dims[0] * scaling), math.ceil(agent_dims[1] * scaling)]
        )
        self.agent_loc = self.transform_xy_to_map(agent_loc[0], agent_loc[1])
        # unit vector for agent direction
        self.agent_dir = agent_dir / np.linalg.norm(agent_dir)
        self.agent_max_linear_speed = agent_max_linear_speed  # in meters/second
        self.agent_max_angular_speed = agent_max_angular_speed  # in radians
        self.set_agent_channel()

        # make the observation space with 3 channels: 1 for the map, 1 for the agent, 1 for spaces visited
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(224, 224, 3),
            dtype=np.uint8,
        )

        # #navigation goal input, 2D vector
        inverted_map = 1 - self.map_channel
        self.astar = Astar(inverted_map, max(self.agent_dims), 0)
        _, self.map_channel_out = self.astar.findable_area(
            self.agent_loc, use_weighted_grid=False
        )
        self.possible_start_positions, self.grid_valid_locations = (
            self.astar.findable_area(self.agent_loc)
        )
        self.coverage_possible = np.count_nonzero(self.map_channel_out)
        self.coverage_channel_out = self.map_channel_out.copy()
        self.curr_coverage = 0
        print("coverage possible: ", self.coverage_possible)
        # action is 8 discrete directions
        self.action_space = spaces.Discrete(8)
        self.coverage_radius = coverage_radius

        self.coverage_weight = 1
        self.time_weight = 0.25

    def reset(self):
        self.coverage_channel = np.zeros(
            (self.image_size_x, self.image_size_y), dtype=np.uint8
        )
        self.coverage_channel_out = self.map_channel_out.copy()
        self.curr_coverage = 0
        # randomly sample a start point
        start_index = np.random.randint(0, len(self.possible_start_positions))
        self.agent_loc = self.possible_start_positions[start_index]
        # randomly sample 2 numbers from -1 to 1 then normalize them
        self.agent_dir = np.random.rand(2) * 2 - 1
        self.agent_dir = self.agent_dir / np.linalg.norm(self.agent_dir)
        self.set_agent_channel()
        return self.get_observation(), {}

    def render(self, mode="human"):
        if mode == "human":
            pygame.init()
            screen = pygame.display.set_mode(
                (self.image_size_x, self.image_size_y), pygame.RESIZABLE
            )
            screen.fill((0, 0, 0))
            # draw the map
            for i in range(self.image_size_x):
                for j in range(self.image_size_y):
                    if self.map_channel[i, j] == 1:
                        pygame.draw.rect(screen, (255, 255, 255), (i, j, 1, 1))
                    if self.coverage_channel[i, j] == 1:
                        pygame.draw.rect(screen, (0, 255, 0), (i, j, 1, 1))
                    if self.agent_channel[i, j] == 1:
                        pygame.draw.rect(screen, (0, 0, 255), (i, j, 1, 1))
            pygame.display.flip()
        elif mode == "rgb_array":
            img = np.zeros((self.image_size_x, self.image_size_y, 3), dtype=np.uint8)
            for i in range(self.image_size_x):
                for j in range(self.image_size_y):
                    if self.map_channel[i, j] == 1:
                        img[i, j] = [255, 255, 255]
                    if self.coverage_channel[i, j] == 1:
                        img[i, j] = [0, 255, 0]
                    if self.agent_channel[i, j] == 1:
                        img[i, j] = [0, 0, 255]
            return img

    def close(self):
        pygame.display.quit()
        pygame.quit()

    def get_observation(self):
        return cv2.resize(
            np.stack(
                [self.map_channel_out, self.agent_channel, self.coverage_channel_out]
            ).transpose(1, 2, 0),
            (224, 224),
        )

    # def get_reward_termination(self):
    #     uncovered = self.coverage_possible - np.count_nonzero(self.coverage_channel)
    #     ratio = uncovered / self.coverage_possible
    #     print("uncovered: ", uncovered, "ratio: ", ratio)
    #     return -self.total_termination_ratio_weight * ratio

    def get_reward(self, total_time, coverage):
        # print("total time: ", total_time, "coverage: ", coverage)
        return (
            coverage / self.scaling * self.coverage_weight
            - total_time * self.time_weight
        )

    def check_next_step(self):
        if not self.is_valid(self.nav_goal[0], self.nav_goal[1]):
            return False
        elif not self.astar.is_unblocked(self.nav_goal[0], self.nav_goal[1]):
            return False
        elif self.grid_valid_locations[self.nav_goal[0], self.nav_goal[1]] == 0:
            return False
        return True

    def step(self, action):
        # check if action is termination
        # add reward for taking the reset action
        # get path from agent to navigation goal
        # 2d vector of tuples
        directions = np.array(
            [
                [0, 1],
                [1, 1],
                [1, 0],
                [1, -1],
                [0, -1],
                [-1, -1],
                [-1, 0],
                [-1, 1],
            ]
        )
        self.nav_goal = (self.agent_loc + directions[action]).squeeze()
        # see if navigation goal is reachable
        if not self.check_next_step():
            return (
                self.get_observation(),
                -1,
                False,
                False,
                {},
            )
        path = np.array([self.agent_loc, self.nav_goal])
        total_time, coverage = self.sweep_path(path)
        self.curr_coverage += coverage
        # self.agent_loc = self.nav_goal[0], self.nav_goal[1]
        self.set_agent_channel()
        # check if agent has reached coverage threshold
        if self.curr_coverage >= self.coverage_possible * self.coverage_required:
            return (
                self.get_observation(),
                25,
                True,
                False,
                {},
            )

        return (
            self.get_observation(),
            self.get_reward(total_time, coverage),
            False,
            False,
            {},
        )

    def get_turn_distance(self, veca, vecb):
        adotb = np.dot(veca, vecb)
        abmag = np.linalg.norm(veca) * np.linalg.norm(vecb)
        return math.acos(np.clip(adotb / abmag, -1, 1))

    def transform_xy_to_map(self, x, y):
        return round((x - self.x_min) * self.scaling), round(
            (y - self.y_min) * self.scaling
        )

    def transform_map_to_xy(self, x, y):
        return x / self.scaling + self.x_min, y / self.scaling + self.y_min

    def set_agent_channel(self):
        self.agent_channel = np.zeros(
            (self.image_size_x, self.image_size_y), dtype=np.uint8
        )
        agent_x, agent_y = self.agent_loc[0], self.agent_loc[1]
        for i in range(
            agent_x - self.agent_dims[0] // 2, agent_x + self.agent_dims[0] // 2
        ):
            for j in range(
                agent_y - self.agent_dims[1] // 2, agent_y + self.agent_dims[1] // 2
            ):
                self.agent_channel[i, j] = 1
        # fill in one square on the perimeter of the map to indicate the direction of the agent

        # # get the center of the map
        # old_x = self.image_size_x // 2
        # old_y = self.image_size_y // 2
        # x = self.image_size_x // 2
        # y = self.image_size_y // 2
        # # move in the direction of the agent until we hit the edge of the map
        # dir = self.agent_dir * self.scaling
        # while self.is_valid(x, y):
        #     x += round(dir[0])
        #     y += round(dir[1])
        # # set the pixel at the edge of the map to 1
        # box_size = max(self.map_padding // 4, 1)
        # if x < 0:
        #     x = box_size
        # elif x >= self.image_size_x:
        #     x = self.image_size_x - 1 - box_size
        # if y < 0:
        #     y = box_size
        # elif y >= self.image_size_y:
        #     y = self.image_size_y - 1 - box_size

        # # draw a box around the pixel of 0.5m
        # for i in range(max(x - box_size, 0), min(x + box_size + 1, self.image_size_x)):
        #     for j in range(
        #         max(y - box_size, 0), min(y + box_size + 1, self.image_size_y)
        #     ):
        #         self.agent_channel[i, j] = 1

    def set_map_channel(self):
        # first channel is the map, which is a binary image where 0 is empty space and 1 is occupied space
        self.map_channel = np.zeros(
            (self.image_size_x, self.image_size_y), dtype=np.uint8
        )
        for line in self.vectormap:
            start_x = round((line["p0"]["x"] - self.x_min) * self.scaling)
            start_y = round((line["p0"]["y"] - self.y_min) * self.scaling)
            end_x = round((line["p1"]["x"] - self.x_min) * self.scaling)
            end_y = round((line["p1"]["y"] - self.y_min) * self.scaling)

            # add padding to the lines
            start_x += self.map_padding
            start_y += self.map_padding
            end_x += self.map_padding
            end_y += self.map_padding

            # horizontal line
            if start_y == end_y:
                start_x, end_x = min(start_x, end_x), max(start_x, end_x)
                self.map_channel[start_x:end_x, start_y] = 1
            # vertical line
            elif start_x == end_x:
                start_y, end_y = min(start_y, end_y), max(start_y, end_y)
                self.map_channel[start_x, start_y:end_y] = 1
            # diagonal line
            else:
                # along x axis
                (start_x, start_y), (end_x, end_y) = sorted(
                    [(start_x, start_y), (end_x, end_y)], key=lambda x: x[0]
                )
                slope = (end_y - start_y) / (end_x - start_x)
                for x in range(start_x, end_x + 1):
                    y = start_y + slope * (x - start_x)
                    self.map_channel[round(x), round(y)] = 1

                # along y axis
                (start_x, start_y), (end_x, end_y) = sorted(
                    [(start_x, start_y), (end_x, end_y)], key=lambda x: x[1]
                )
                slope = (end_x - start_x) / (end_y - start_y)
                for y in range(start_y, end_y + 1):
                    x = start_x + slope * (y - start_y)
                    self.map_channel[round(x), round(y)] = 1

    def is_valid(self, x, y):
        return 0 <= x < self.map_channel.shape[0] and 0 <= y < self.map_channel.shape[1]

    def line_coverage_channel_sweep(
        self,
        p0,
        p1,
        increment=0.5,
    ):
        # increment is equal to the value of 1 with regards to the scaling factor, so if the scaling factor is 10, increment is 0.5*1/10 = 0.05 meters
        # this function increments along a path line, and draws in the coverage radius by extending lines perpendicular to the path line.
        # the extended lines also increment and set the coverage channel to 1. This happens all along the path line giving us where the robot has covered.
        # dz = increment / scaling
        radius_scaled = self.coverage_radius * self.scaling
        segment_length = np.linalg.norm(p1 - p0)
        # segment length now unscaled
        segment_length_scaled = segment_length  # *scaling
        # m = (p1[1] - p0[1])/(p1[0] - p0[0])
        # m_inv = -1/m
        theta = math.atan2((p1[1] - p0[1]), (p1[0] - p0[0]))
        tangent_line_theta = math.atan2(-(p1[0] - p0[0]), (p1[1] - p0[1]))
        d_coverage_radius = round(radius_scaled / increment)
        d_line_length = round(segment_length_scaled / increment)
        # position along the main line
        # pos = [(p0[0]-x_min)*10, (p0[1]-y_min)*10] #scaled version for original values
        pos = [p0[0], p0[1]]  # assumed scaling done before this function
        heading_rad_inc = [
            increment * math.cos(tangent_line_theta),
            increment * math.sin(tangent_line_theta),
        ]
        heading_segment_inc = [increment * math.cos(theta), increment * math.sin(theta)]

        # reward function calculations here, percent of recovered areas compared to uncovered areas being covered.
        new_coverage = 0
        # recoverage = 0
        for d_ll in range(d_line_length + 1):
            old_rad_pos = [0, 0]
            old_rad_neg = [0, 0]
            rad_pos = pos
            rad_neg = pos
            stop_pos = False
            stop_neg = False
            for d_cr in range(d_coverage_radius + 1):
                if (
                    self.is_valid(round(rad_pos[0]), round(rad_pos[1]))
                    and self.astar.coverage_grid[round(rad_pos[0]), round(rad_pos[1])]
                    != float("inf")
                    and not stop_pos
                ):

                    # set the coverage channel to 1
                    self.coverage_channel[round(rad_pos[0]), round(rad_pos[1])] = 1
                    self.coverage_channel_out[round(rad_pos[0]), round(rad_pos[1])] = 0
                    # calculate the pre-requisite for the reward function
                    # assign the new position
                    rad_pos = [
                        rad_pos[0] + heading_rad_inc[0],
                        rad_pos[1] + heading_rad_inc[1],
                    ]
                else:
                    stop_pos = True
                if (
                    self.is_valid(round(rad_neg[0]), round(rad_neg[1]))
                    and self.astar.coverage_grid[round(rad_neg[0]), round(rad_neg[1])]
                    != float("inf")
                    and not stop_neg
                ):

                    # same as above but for the negative direction

                    self.coverage_channel[round(rad_neg[0]), round(rad_neg[1])] = 1
                    self.coverage_channel_out[round(rad_neg[0]), round(rad_neg[1])] = 0

                    rad_neg = [
                        rad_neg[0] - heading_rad_inc[0],
                        rad_neg[1] - heading_rad_inc[1],
                    ]
                else:
                    stop_neg = True
            pos = [pos[0] + heading_segment_inc[0], pos[1] + heading_segment_inc[1]]
        # print("new coverage: ", new_coverage, "recoverage: ", recoverage)
        # the circular area it covers at the end of a line segment, the circle is fucked up and could be fixed but its good enough
        for d_circ in range(d_coverage_radius + 1):
            old_rad_pos = [0, 0]
            old_rad_neg = [0, 0]
            rad_pos = pos
            rad_neg = pos
            if not self.is_valid(
                round(rad_pos[0]), round(rad_pos[1])
            ) or self.astar.coverage_grid[
                round(rad_pos[0]), round(rad_pos[1])
            ] == float(
                "inf"
            ):
                return segment_length
            stop_pos = False
            stop_neg = False
            d_coverage_circle = round(
                math.sqrt(max(d_coverage_radius**2 - (d_circ * increment) ** 2, 0))
            )

            for d_cr in range(d_coverage_circle + 1):
                if (
                    self.is_valid(round(rad_pos[0]), round(rad_pos[1]))
                    and self.astar.coverage_grid[round(rad_pos[0]), round(rad_pos[1])]
                    != float("inf")
                    and not stop_pos
                ):

                    self.coverage_channel[round(rad_pos[0]), round(rad_pos[1])] = 1
                    self.coverage_channel_out[round(rad_pos[0]), round(rad_pos[1])] = 0

                    rad_pos = [
                        rad_pos[0] + heading_rad_inc[0],
                        rad_pos[1] + heading_rad_inc[1],
                    ]
                else:
                    stop_pos = True
                if (
                    self.is_valid(round(rad_neg[0]), round(rad_neg[1]))
                    and self.astar.coverage_grid[round(rad_neg[0]), round(rad_neg[1])]
                    != float("inf")
                    and not stop_neg
                ):
                    round_rad_neg = [round(rad_neg[0]), round(rad_neg[1])]
                    self.coverage_channel[round(rad_neg[0]), round(rad_neg[1])] = 1
                    self.coverage_channel_out[round(rad_neg[0]), round(rad_neg[1])] = 0

                    rad_neg = [
                        rad_neg[0] - heading_rad_inc[0],
                        rad_neg[1] - heading_rad_inc[1],
                    ]
                else:
                    stop_neg = True
            pos = [pos[0] + heading_segment_inc[0], pos[1] + heading_segment_inc[1]]
        # print("new coverage: ", new_coverage, "recoverage: ", recoverage)
        return segment_length

    # this function is used to sweep the path and update the coverage channel, it returns coverage channel, how many cells were newly covered, and how many were gone over again (recoverage)
    def sweep_path(self, path, increment=0.5):
        p0 = path[0]
        old_coverage = np.count_nonzero(self.coverage_channel)
        # total_length = 0
        total_time = 0
        for p1 in path[1:]:
            segment_length = self.line_coverage_channel_sweep(p0, p1, increment)
            p0_p1_vec = (p1 - p0) / np.linalg.norm(p1 - p0)
            turn_distance = self.get_turn_distance(self.agent_dir, p0_p1_vec)
            turn_time = turn_distance / self.agent_max_angular_speed
            linear_time = segment_length / self.scaling / self.agent_max_linear_speed
            self.agent_dir = p0_p1_vec
            self.agent_loc = p1
            p0 = p1
            # total_length += segment_length
            total_time += turn_time + linear_time
        new_coverage = np.count_nonzero(self.coverage_channel) - old_coverage
        return total_time, new_coverage