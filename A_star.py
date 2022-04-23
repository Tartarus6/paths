import numpy
from random import random, randint
import matplotlib.pyplot as plt
from time import time
from matplotlib.animation import FuncAnimation
# note: "node" refers to a position on the maze
# note: the value of a node is the cost that it takes to get there, so lower cost is better
# node: any code with "# non-resilient", and probably some more, will need to be changed for any major change in the pathfinding such as adding walls or changing the maze shape

def grid_print(grid):  # non-resilient
    for i in grid:
        print(i)


def unzip(lst):  # yields the contents of the input list, removing nested lists. Output needs to be converted into a list to be used.
    for i in lst:
        if type(i) is list:
            yield from list(unzip(i))
        else:
            yield i


def make_maze(x: int, y: int, value_range: tuple, wall_rate: float):
    maze = [[0 for i in range(x)] for j in range(y)]  # initialising array of zeros

    # adding values to the array
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            maze[i][j] = randint(value_range[0], value_range[1])  # adding cost values
            if random() < wall_rate:
                maze[i][j] = -1

    return maze


def a_star(maze, starting_position: tuple, goal_position: tuple):
    print("run")
    
    # maze start and end can't be walls
    maze[starting_position[0]][starting_position[1]] = 10
    maze[goal_position[0]][goal_position[1]] = 10

    
    def heuristic(current_position, end_position):  # estimates the cost to reach goal from the node at the given position
        # non-resilient
        
        
        # get average value of a node
        unzipped_maze = list(unzip(maze))
        average_value = sum(list(unzipped_maze)) / len(unzipped_maze)
        
        # estimating moves needed to get to goal
        # assumes no walls or obstructions
        distance_to_goal = abs(max(current_position[0], end_position[0]) - min(current_position[0], end_position[0])) + abs(max(current_position[1], end_position[1]) - min(current_position[1], end_position[1]))
        
        return distance_to_goal * average_value
    
    def neighbors(position):  # finding available positions to move into from the given position
        # non-resilient
        
        output = []

        for i in range(-1, 2):  # y offset  # non-resilient
            for j in range(-1, 2):  # x offset  # non-resilient
                if i == 0 and j == 0:  # makes it so that the input position won't be added to the output as one of the neighbors and won't be outside the maze  # non-resilient
                    continue
                if (position[0] + i < 0) or (position[1] + j < 0) or (position[0] + i >= len(maze)) or (position[1] + j >= len(maze[i])):  # makes it so that the output won't be outside the maze
                    continue
                if (get_value((position[0] + i, position[1] + j)) == -1):  # if the neighbor is a wall
                    continue

                output.append((position[0] + i, position[1] + j))

        return output

    def get_value(position):  # returns the value of the input position in the maze
        # non-resilient
        
        return maze[position[0]][position[1]]

    infinity = float('inf')  # useful for making future code more readable
    
    # list of Nodes that need to be explored
    ToDo = [starting_position]
    
    # gScore[n] on wiklipedia will be shortest_path_cost[n] on here
    shortest_path_cost = {}
    # set all position's shortest path cost value to infinity
    for i in range(len(maze)):  # non-resilient
        for j in range(len(maze[i])):  # non-resilient
            shortest_path_cost[(i, j)] = infinity  # non-resilient
    shortest_path_cost[starting_position] = get_value(starting_position)  # the cost to reach the starting position will be 0
    
    # came_from[n] is the neighboring node directly preceding n on the shortest path to n
    came_from = {}
    
    # heuristic_dict[n] gives the output of heuristic(n) without having to re-run it
    # probably wouldn't be necessary in a maze like project euler problem 67
    # heuristic is used to prioritise movement towards the end position
    heuristic_dict = {}  # non-resilient
    for i in range(len(maze)):  # non-resilient
        for j in range(len(maze[i])):  # non-resilient
            heuristic_dict[(i, j)] = heuristic((i, j), goal_position)

    # For node n, to_finish_through[n] := gScore[n] + h(n).fScore[n] represents our current best guess as to
    # how short a path from start to finish can be if it goes through n.
    to_finish_through = {}
    # set all position's path length value to infinity
    for i in range(len(maze)):  # non-resilient
        for j in range(len(maze[i])):  # non-resilient
            to_finish_through[(i, j)] = infinity  # non-resilient
    
    
    def reconstruct_path(came_from: dict, current: tuple):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.insert(0, current)
        return path

    while len(ToDo) > 0:  # while there are still ToDo nodes
        value_dict = {pos: maze[pos[0]][pos[1]] for pos in ToDo}  # the values of each position on the ToDo list
        current_node = min(ToDo, key=heuristic_dict.get)  # get the position with the lowest heuristic value that's in the ToDo list

        if current_node == goal_position:  # if the current node is the destination, end
            return reconstruct_path(came_from=came_from, current=current_node), maze

        ToDo.remove(current_node)  # node has been "explored", so remove it from the ToDo list

        for neighbor in neighbors(current_node):
            # tentative_shortest_path_cost is the distance from start to the neighbor through current
            
            try:
                tentative_shortest_path_cost = shortest_path_cost[current_node] + get_value(neighbor)
            except IndexError as e:
                print(f"error: {e} \ncurrent node: {current_node} \nneighbor node: {neighbor}")

            if tentative_shortest_path_cost < shortest_path_cost[neighbor]:  # if the new path to neighbor is the shortest path to neighbor
                came_from[neighbor] = current_node

                shortest_path_cost[neighbor] = tentative_shortest_path_cost
                heuristic_dict[neighbor] = tentative_shortest_path_cost + heuristic(neighbor, goal_position)
                
                if neighbor not in ToDo:
                    ToDo.append(neighbor)
    
    
    return "FAILED!!!     Change this to something better later", maze

    """Note: Change this to something better later"""


def animate_path(maze, path):
    if type(path) == str:  # that being a string indicates that the algorithm failed to find a path
        return
    
    fig, ax = plt.subplots()
    
    display_array = numpy.array(maze)
    for position in path:
        display_array[position[0], position[1]] = 5
    graph = plt.imshow(display_array)
    
    plt.show()


start = time()


maze = make_maze(x=50, y=50, value_range=(10, 10), wall_rate=0.3)


output = a_star(maze=maze, starting_position=(0, 0), goal_position=(len(maze)-1, len(maze[0])-1))  # non-resilient
print(f"path: {output[0]}")
print("grid:")
grid_print(output[1])

elapsed = time() - start

print(f"elapsed: {elapsed}s")

animate_path(maze=output[1], path=output[0])
