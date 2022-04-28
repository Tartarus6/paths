import numpy
from random import random, randint
import matplotlib.pyplot as plt
from time import time, sleep
from matplotlib.animation import FuncAnimation
import imageio
# note: "node" refers to a position on the maze
# note: the value of a node is the cost that it takes to get there, so lower cost is better
# note: any code with "# non-resilient", and probably some more, will need to be changed for any major change in the pathfinding such as adding walls or changing the maze shape


# written based off of the pseudocode at "https://en.wikipedia.org/wiki/A*_search_algorithm"


def make_gif(frame_folder, fps, num_frames):
    print("gif")
    
    giffile = 'gif.gif'
    
    images_data = []
    for i in range(num_frames):
        data = imageio.imread(f'gif_folder/frame_{i}.jpg')
        images_data.append(data)
    
    imageio.mimwrite(giffile, images_data, format='.gif', fps=fps)


def grid_print(grid):  # non-resilient
    for i in grid:
        print(i)


def unzip(lst):  # yields the contents of the input list, removing nested lists. Output needs to be converted into a list to be used.
    for i in lst:
        if type(i) is list:
            yield from list(unzip(i))
        else:
            yield i


def make_maze(x: int, y: int, value_range: int, wall_rate: float):
    maze = [[0 for i in range(x)] for j in range(y)]  # initialising array of zeros

    # adding values to the array
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            maze[i][j] = randint(10, 10 + value_range)  # adding cost values
            if random() < wall_rate:
                maze[i][j] = -1

    return maze


def a_star(maze, starting_position: tuple, goal_position: tuple):
    print("run")
    
    # maze start and end can't be walls
    maze[starting_position[0]][starting_position[1]] = 10
    maze[goal_position[0]][goal_position[1]] = 10
    
    
    class Output:
        def __init__(self, maze=None, final_path=None, failed=False, to_do_history=None, path_history=None, explored_history=None):
            if to_do_history is None:
                to_do_history = []
            self.maze = maze
            self.final_path = final_path
            self.failed = failed
            self.to_do_history = to_do_history
            self.path_history = path_history
            self.explored_history = explored_history
            

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
                if get_value((position[0] + i, position[1] + j)) == -1:  # if the neighbor is a wall
                    continue

                output.append((position[0] + i, position[1] + j))

        return output

    def get_value(position):  # returns the value of the input position in the maze
        # non-resilient
        
        return maze[position[0]][position[1]]

    infinity = float('inf')  # useful for making future code more readable
    
    # list of Nodes that need to be explored
    to_do = [starting_position]
    
    # the complete history of the to_do list, not used in algorithm, just for visual output
    to_do_history = []

    # list of Nodes that have been explored
    explored = [starting_position]

    # the complete history of the explored list, not used in algorithm, just for visual output
    explored_history = []
    
    # list of the paths taken to explore nodes, not used in algorithm, just for visual output
    path_history = []
    
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

    while len(to_do) > 0:  # while there are still to_do nodes
        value_dict = {pos: maze[pos[0]][pos[1]] for pos in to_do}  # the values of each position on the to_do list
        current_node = min(to_do, key=heuristic_dict.get)  # get the position with the lowest heuristic value that's in the to_do list
        
        

        if current_node == goal_position:  # if the current node is the destination, end
            return Output(final_path=reconstruct_path(came_from=came_from, current=current_node), maze=maze, to_do_history=to_do_history, path_history=path_history, explored_history=explored_history)

        to_do.remove(current_node)  # node has been "explored", so remove it from the to_do list
        explored.append(current_node)
        
        to_do_history.append(list(to_do))  # add the current to_do list to the to_do_history list, the list function is run so that the element in to_do_history won't update when to_do updates
        path_history.append(reconstruct_path(came_from=came_from, current=current_node))  # adds the path taken to the current node to path_history
        explored_history.append(list(explored))

        for neighbor in neighbors(current_node):
            # tentative_shortest_path_cost is the distance from start to the neighbor through current
            
            tentative_shortest_path_cost = shortest_path_cost[current_node] + get_value(neighbor)
            if tentative_shortest_path_cost < shortest_path_cost[neighbor]:  # if the new path to neighbor is the shortest path to neighbor
                came_from[neighbor] = current_node

                shortest_path_cost[neighbor] = tentative_shortest_path_cost
                heuristic_dict[neighbor] = tentative_shortest_path_cost + heuristic(neighbor, goal_position)
                
                if neighbor not in to_do:
                    to_do.append(neighbor)
    
    
    return Output(failed=True)


def animate_path(show_final_path_frames: int, maze: list, path_color: list, explored_color: list, to_do_color: list, final_path: list, path_history: list, to_do_history: list, explored_history: list, fps: float, save_files=False):
    if type(final_path) == str:  # if final_path is a string
        return  # that being a string indicates that the algorithm failed to find a path, exit the function
    
    if len(path_history) != len(to_do_history):  # if the lengths of ant of the history lists are different
        return  # something went wrong, just exit

    fig, ax = plt.subplots()  # initialising plot

    display_array = numpy.array(maze)

    graph = plt.imshow(display_array)  # add the array to the window


    # color the maze
    max_value_in_maze = max(unzip(maze))
    min_value_in_maze = min(unzip(maze))
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            current_value = maze[i][j]

            if maze[i][j] == -1:
                maze[i][j] = [0, 0, 0]
            else:
                maze[i][j] = (current_value - min_value_in_maze) / (max_value_in_maze - min_value_in_maze)

                maze[i][j] = int(maze[i][j] * 128 + 127)
                maze[i][j] = [maze[i][j], maze[i][j], maze[i][j]]

    def animate(i):
        display_array = numpy.array(maze)  # reset display_array

        if i >= len(path_history):  # if it's on the last frame of the animation
            # show the final path
            for node in final_path:  # for each node in the final path
                display_array[node] = path_color
            # hold the animation on the last frame so that the final path is visible
            pass
            

            graph.set_data(display_array)
            
            # save frame
            if save_files:
                plt.savefig(f"gif_folder/frame_{i}.jpg")
                if i == len(path_history) + show_final_path_frames - 1:  # if it's on the final frame of the animation
                    make_gif("gif_folder", fps=fps, num_frames=i+1)
            return fig
            
        
        # pos = (1, 1)
        # display_array[pos] = display_array[pos] - 1
        for node in explored_history[i]:  # for each explored node on the frame 'i'
            display_array[node] = explored_color
        for node in to_do_history[i]:  # for each node in to_do on frame 'i'
            display_array[node] = to_do_color
        for node in path_history[i]:  # for each node in the path of frame 'i'
            display_array[node] = path_color
        
        
        graph.set_data(display_array)
        
        # save frame
        if save_files:
            plt.savefig(f"gif_folder/frame_{i}.jpg")
        return fig
    

    ani = FuncAnimation(fig, animate, frames=len(path_history) + show_final_path_frames, interval=1000//fps)
    plt.show()


def calculate_path_cost(maze, path):
    # non-resilient
    total_cost = 0
    for position in path:
        total_cost += maze[position[0]][position[1]]
    return total_cost


# arguments ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼

maze_x = 25
maze_y = 25
maze_value_range = 0  # int showing the amount of different values that can appear on the maze
maze_wall_rate = 0.4

starting_position = (0, 0)
goal_position = (maze_y-1, maze_x-1)

animation_save_files = True
animation_show_final_path_frames = 10
animation_fps = 45
animation_path_color = [238, 255, 13]
animation_explored_color = [85, 208, 230]
animation_to_do_color = [115, 227, 113]

# arguments ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


start = time()


maze = make_maze(x=maze_x, y=maze_y, value_range=maze_value_range, wall_rate=maze_wall_rate)


"""maze = [[10, 10, 10, 10, 10, 10, 10],
        [10, 10, 10, 10, 10, -1, 10],
        [10, 10, 10, 10, 10, -1, 10],
        [10, 10, 10, 10, 10, -1, 10],
        [10, 10, 10, 10, 10, -1, 10],
        [10, -1, -1, -1, -1, -1, 10],
        [10, 10, 10, 10, 10, 10, 10]]"""


output = a_star(maze=maze, starting_position=starting_position, goal_position=goal_position)  # non-resilient

elapsed = time() - start



if not output.failed:
    print(f"elapsed: {elapsed}s")
    
    animate_path(maze=output.maze, path_color=animation_path_color, explored_color=animation_explored_color, to_do_color=animation_to_do_color, final_path=output.final_path, path_history=output.path_history, to_do_history=output.to_do_history, explored_history=output.explored_history, save_files=animation_save_files, show_final_path_frames=animation_show_final_path_frames, fps=animation_fps)
else:
    print("Pathfinding failed :(")
