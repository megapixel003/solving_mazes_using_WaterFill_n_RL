import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from collections import deque

# Maze parameters
SIZE = 51
start = (SIZE // 2, SIZE // 2)

goal = (1, 1)

# Generate maze using DFS
def generate_maze(size, extra_open=150):
    maze = np.ones((size, size), dtype=int)

    def carve(x, y):
        maze[x, y] = 0
        dirs = [(0,2), (2,0), (0,-2), (-2,0)]
        random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 < nx < size-1 and 0 < ny < size-1 and maze[nx, ny] == 1:
                maze[x + dx // 2, y + dy // 2] = 0
                carve(nx, ny)

    carve(start[0], start[1])
    maze[goal] = 0

    # Open random walls to add complexity (controlled)
    max_open_ratio = 0.2  # không mở quá 20% số tường
    walls = [(i, j) for i in range(1, size-1) for j in range(1, size-1) if maze[i, j] == 1]
    random.shuffle(walls)

    max_openable = int(len(walls) * max_open_ratio)
    open_count = min(extra_open, max_openable)

    for i in range(open_count):
        x, y = walls[i]
        maze[x, y] = 0

    return maze

maze = generate_maze(SIZE)
rows, cols = maze.shape

# Water-fill and return both steps & distance map
def water_fill_animation(maze, goal):
    dist = np.full_like(maze, -1)
    q = deque([goal])
    dist[goal] = 0
    steps = [[goal]]

    while q:
        current_level = []
        for _ in range(len(q)):
            x, y = q.popleft()
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and maze[nx, ny] == 0 and dist[nx, ny] == -1:
                    dist[nx, ny] = dist[x, y] + 1
                    q.append((nx, ny))
                    current_level.append((nx, ny))
        if current_level:
            steps.append(current_level)
    return steps, dist

# Trace shortest path using distance map
def trace_shortest_path(start, goal, dist):
    path = []
    x, y = start
    if dist[x, y] == -1:
        return []  # No path
    path.append((x, y))
    while (x, y) != goal:
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and dist[nx, ny] == dist[x, y] - 1:
                x, y = nx, ny
                path.append((x, y))
                break
    return path

# Generate water steps and distance map
water_steps, dist_map = water_fill_animation(maze, goal)
shortest_path = trace_shortest_path(start, goal, dist_map)

# Visualization
fig, ax = plt.subplots(figsize=(8, 8))
maze_rgb = np.stack([maze] * 3, axis=-1).astype(float)
maze_rgb[maze == 1] = [0, 0, 0]    # walls
maze_rgb[maze == 0] = [1, 1, 1]    # paths

img = ax.imshow(maze_rgb)
ax.set_title("Water Fill Simulation")

# Animation step
def update(frame):
    if frame < len(water_steps):
        for x, y in water_steps[frame]:
            maze_rgb[x, y] = [0.2, 0.6, 1.0]  # water color (light blue)
    elif frame == len(water_steps):  # After water fill is complete
        for x, y in shortest_path:
            maze_rgb[x, y] = [1.0, 0.2, 0.2]  # shortest path: red
    img.set_data(maze_rgb)
    ax.set_title(f"Frame {frame}")
    return img,

# Total frames = water steps + 1 for shortest path
ani = animation.FuncAnimation(fig, update, frames=len(water_steps)+1, interval=50, repeat=False)
plt.show()
