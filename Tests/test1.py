import numpy
import matplotlib.pyplot as plt
import matplotlib.animation as animation


fig, ax = plt.subplots()  # initialising plot

maze = [[10, 10, 10, 10, 10, 10],
        [10, 10, 10, 10, -1, 10],
        [10, 10, 10, 10, -1, 10],
        [10, 10, 10, 10, -1, 10],
        [10, -1, -1, -1, -1, 10],
        [10, 10, 10, 10, 10, 10]]

display_array = numpy.array(maze)

graph = plt.imshow(display_array)  # add the array to the window


def animate(i):
    pos = (1, 1)
    display_array[pos] = display_array[pos] - 1
    graph.set_data(display_array)
    return fig


ani = animation.FuncAnimation(fig, animate, frames=100, interval=50)
plt.show()
