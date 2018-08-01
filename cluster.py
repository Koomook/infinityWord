from collections import defaultdict

points = [(1.0, 2.0),
          (1.0, 3.0),
          (2.0, 2.7),
          (2.4, 1.7),
          (2.7, 4.5),
          (3.0, 1.0),
          (4.1, 2.2),
          (4.7, 4.6),
          (4.8, 1.1),
          (4.9, 2.3)]

points_indexed = [(i, x, y) for i, (x,y) in enumerate(points)]

K = 3

max_x = max([x for x, y in points]) + 0.01
min_x = min([x for x, y in points]) - 0.01
max_y = max([y for x, y in points]) + 0.01
min_y = min([y for x, y in points]) - 0.01

START_NUM_GRIDS = len(points)
num_girds = START_NUM_GRIDS
grid_width = (max_x - min_x) / num_girds
grid_height = (max_y - min_y) / num_girds

results = []
cluster_index = 0

while True:
    grid = defaultdict(lambda: defaultdict(list))
    remaining_points = []
    for point_index, x, y in points_indexed:
        grid_i = int((x - min_x) // grid_width)
        grid_j = int((y - min_y) // grid_height)

        grid_elements = grid[grid_i][grid_j]
        if len(grid_elements) < K:
            grid[grid_i][grid_j].append((point_index, x, y))
            remaining_points.append((point_index, x, y))
        else:
            points_cluster_annotated = [(point_index, x, y, cluster_index)
                                        for point_index, x, y in grid[grid_i][grid_j]]
            results.append(points_cluster_annotated)
            grid[grid_i][grid_j] = []  # empty grid
            cluster_index += 1

    num_girds = num_girds // 2 + 1 if num_girds > 2 else num_girds // 2
    if num_girds == 0: break

    grid_width = (max_x - min_x) / num_girds
    grid_height = (max_y - min_y) / num_girds
    points_indexed = remaining_points

print(results)
