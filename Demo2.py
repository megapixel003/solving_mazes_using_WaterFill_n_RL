import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from collections import deque

# Maze parameters
SIZE = 41  # Size of the maze (must be odd)
start = (1, 1)
goal = (SIZE//2, SIZE//2)

# Maze generation using DFS
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

    # Open some walls to add complexity
    max_open_ratio = 0.1
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

# Water-filling cost map
def water_fill(maze, goal):
    dist = np.full_like(maze, -1)
    q = deque([goal])
    dist[goal] = 0
    while q:
        x, y = q.popleft()
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx, ny] == 0 and dist[nx, ny] == -1:
                dist[nx, ny] = dist[x, y] + 1
                q.append((nx, ny))
    return dist

cost_map = water_fill(maze, goal)

# Q-learning
Q = np.zeros((rows, cols, 4))
alpha, gamma, epsilon = 0.1, 0.9, 0.2
actions = [(-1,0),(1,0),(0,-1),(0,1)]

def choose_action(x, y):
    if random.random() < epsilon:
        return random.randint(0, 3)
    return np.argmax(Q[x, y])

def step(x, y, action):
    dx, dy = actions[action]
    nx, ny = x + dx, y + dy
    if not (0 <= nx < rows and 0 <= ny < cols) or maze[nx, ny] == 1:
        return x, y, -10
    if (nx, ny) == goal:
        return nx, ny, 100
    shaping = -cost_map[nx, ny] if cost_map[nx, ny] > 0 else -1
    return nx, ny, -1 + shaping * 0.05

# Train Q-learning agent
agent_path = []
for episode in range(1000):
    x, y = start
    steps = []
    for _ in range(1000):
        a = choose_action(x, y)
        nx, ny, r = step(x, y, a)
        Q[x, y, a] += alpha * (r + gamma * np.max(Q[nx, ny]) - Q[x, y, a])
        x, y = nx, ny
        steps.append((x, y))
        if (x, y) == goal:
            break
    if episode == 999:
        agent_path = steps

# Visualization with color
fig, ax = plt.subplots(figsize=(8, 8))

# Create RGB maze image
maze_rgb = np.stack([maze]*3, axis=-1).astype(float)
maze_rgb[maze == 1] = [0, 0, 0]   # wall: black
maze_rgb[maze == 0] = [1, 1, 1]   # path: white

img = ax.imshow(maze_rgb)
agent_dot, = ax.plot([], [], 'ro', markersize=4)

def update(frame):
    if frame > 0:
        x, y = agent_path[frame - 1]
        maze_rgb[x, y] = [0.2, 0.4, 1.0]  # blue path
    x, y = agent_path[frame]
    agent_dot.set_data([y], [x])
    ax.set_title(f"Step {frame}")
    img.set_data(maze_rgb)
    return img, agent_dot

ani = animation.FuncAnimation(fig, update, frames=len(agent_path), interval=30, repeat=False)
plt.show()
