maze = [[1, 2, 4, 7, 1, 6],
        [3, 5, 8, 2, 7, 2],
        [6, 9, 3, 8, 3, 7],
        [0, 4, 9, 4, 8, 1],
        [5, 0, 5, 9, 2, 4],
        [1, 6, 0, 3, 5, 6]]

path = [(0, 0), (1, 0), (2, 0), (3, 0)]


def path_cost(maze, path):
    total = 0
    for i in path:
        total += maze[i[0]][i[1]]
    return total


print(path_cost(maze, path))
