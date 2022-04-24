import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from random import randint, random


# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes()


def make_maze(x: int, y: int, value_range: tuple, wall_rate: float):
    maze = [[0 for i in range(x)] for j in range(y)]  # initialising array of zeros

    # adding values to the array
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            maze[i][j] = randint(value_range[0], value_range[1])  # adding cost values
            if random() < wall_rate:
                maze[i][j] = -1

    return maze


maze = make_maze(x=10, y=10, value_range=(10, 10), wall_rate=0.2)

a = np.array(maze)
im = plt.imshow(a)


def animate(i):
    global maze
    maze[1][1] = maze[1][1] - 1
    print(maze)
    im.set_array(np.array(maze))
    return im


ani = animation.FuncAnimation(fig, animate, frames=100, interval=500)
plt.show()
