from copy import copy


def grid_print(grid):  # non-resilient
    for i in grid:
        print(i)
    print()


maze = [[2, 2],
        [2, 2]]


def full_map(grid):
	full = []
	
	temp_map = []
	for i in range(len(grid)):
		temp_map.append(copy(grid[i]))
		full.append(copy(grid[i]))
	
	def increment_map(map):
		for i in range(len(map)):
			for j in range(len(map[i])):
				map[i][j] = map[i][j] + 1
				if map[i][j] == 10:
					map[i][j] = 1
		
		return map
	
	temp_map = []
	for i in range(len(grid)):
		temp_map.append(copy(grid[i]))
	
	for i in range(1, 5):
		temp_map = increment_map(temp_map)
		for j in range(len(temp_map)):
			for k in range(len(temp_map[j])):
				full[j].append(temp_map[k][j])
	
	temp_map = []
	for i in full:
		temp_map.append(copy(i))
	
	for i in range(1, 5):
		temp_map = increment_map(temp_map)
		for j in temp_map:
			full.append(copy(j))
	
	return full


new = full_map(maze)

print('\n\n\n\n')
print(f'final grid:')
grid_print(new)
