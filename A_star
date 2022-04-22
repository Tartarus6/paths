import numpy as np
from random import random, randint

# note: "node" refers to a position on the maze
# note: the value of a node is the cost that it takes to get there, so lower cost is better

def grid_print(grid):
    for i in grid:
        print(i)


def make_maze(x: int, y: int, value_range: tuple):
    maze = [[0 for i in range(x)] for j in range(y)]  # initialising array of zeros

    # adding values to the array
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            maze[i][j] = randint(value_range[0], value_range[1])  # adding cost values

    return maze


maze = make_maze(x=10, y=10, value_range=(0, 9))

grid_print(maze)


def a_star(maze, starting_position: tuple, goal_position: tuple):
    def neighbors(position):
        output = []

        for i in range(-1, 2):  # y offset
            for j in range(-1, 2):  # x offset
                if i == 0 and j == 0:  # makes it so that the input position won't be added to the output as one of the neighbors
                    continue

                output.append((position[0] + i, position[1] + j))

        return output

    def get_value(position):
        return maze[position[0]][position[1]]

    open = [starting_position]
    closed = []

    # gScore[n] on wiklipedia will be shortest_path[n] on here
    shortest_path = {starting_position: 0}

    # came_from[n] is the neighboring node directly preceding n on the shortest path to n
    came_from = {}

    while len(open) > 0:  # while there are still opened nodes
        value_dict = {pos: maze[pos[0]][pos[1]] for pos in opened}  # the values of each position on the open list
        current_node = min(value_dict, key=value_dict.get)  # get the position with the lowest value

        if current_node == goal_position:  # if the current node is the destination, end
            return "DONE!!!     Change this to something that returns the path"

            """Note: Change this to something that returns the path"""

        open.remove(current_node)  # node has been "explored", so remove it from the open list

        for neighbor in neighbors(current_node):
            # tentative_shortest_path is the distance from start to the neighbor through current

            tentative_shortest_path = shortest_path[current_node] + get_value(neighbor)

            if neighbor not in shortest_path:  # if a path to neighbor hasn't been found yet
                shortest_path[neighbor] = tentative_shortest_path
            elif tentative_shortest_path < shortest_path[neighbor]:  # if the new path to neighbor is the shortest path to neighbor
                came_from[neighbor] = current_node
                pass
