import numpy as np
import heapq
from collections import deque

def to_tuple(x):
    """Convierte numpy, lista o tupla en coordenada (r, c)."""
    if isinstance(x, (list, tuple)):
        return (int(x[0]), int(x[1]))
    if isinstance(x, np.ndarray):
        return (int(x[0]), int(x[1]))
    raise TypeError(f"Coordenada inválida: {x}")

class PathPlanner:
    DIRS = [(1,0), (-1,0), (0,1), (0,-1)]

    def __init__(self, grid):
        self.grid = grid
        self.H, self.W = grid.shape

    def in_bounds(self, r, c):
        return 0 <= r < self.H and 0 <= c < self.W

    def is_free(self, r, c):
        return self.grid[r, c] == 0

    def reconstruct(self, came_from, start, goal):
        if goal not in came_from:
            return []
        path = [goal]
        cur = goal
        while cur != start:
            cur = came_from[cur]
            path.append(cur)
        return path[::-1]

    # ---------------------------------------------------------
    # BFS
    # ---------------------------------------------------------
    def bfs(self, start, goal):
        start = to_tuple(start)
        goal = to_tuple(goal)

        queue = deque([start])
        came_from = {start: None}

        while queue:
            r, c = queue.popleft()

            if (r, c) == goal:
                return self.reconstruct(came_from, start, goal)

            for dr, dc in self.DIRS:
                nr, nc = r + dr, c + dc
                if self.in_bounds(nr, nc) and self.is_free(nr, nc):
                    if (nr, nc) not in came_from:
                        came_from[(nr, nc)] = (r, c)
                        queue.append((nr, nc))

        return []

    # ---------------------------------------------------------
    # DFS
    # ---------------------------------------------------------
    def dfs(self, start, goal):
        start = to_tuple(start)
        goal = to_tuple(goal)

        stack = [start]
        came_from = {start: None}

        while stack:
            r, c = stack.pop()

            if (r, c) == goal:
                return self.reconstruct(came_from, start, goal)

            for dr, dc in self.DIRS:
                nr, nc = r + dr, c + dc
                if self.in_bounds(nr, nc) and self.is_free(nr, nc):
                    if (nr, nc) not in came_from:
                        came_from[(nr, nc)] = (r, c)
                        stack.append((nr, nc))

        return []

    # ---------------------------------------------------------
    # Dijkstra
    # ---------------------------------------------------------
    def dijkstra(self, start, goal):
        start = to_tuple(start)
        goal = to_tuple(goal)

        pq = [(0, start)]
        came_from = {start: None}
        cost = {start: 0}

        while pq:
            cur_cost, (r, c) = heapq.heappop(pq)

            if (r, c) == goal:
                return self.reconstruct(came_from, start, goal)

            for dr, dc in self.DIRS:
                nr, nc = r + dr, c + dc
                if self.in_bounds(nr, nc) and self.is_free(nr, nc):
                    new_cost = cur_cost + 1
                    if (nr, nc) not in cost or new_cost < cost[(nr, nc)]:
                        cost[(nr, nc)] = new_cost
                        came_from[(nr, nc)] = (r, c)
                        heapq.heappush(pq, (new_cost, (nr, nc)))

        return []

    # ---------------------------------------------------------
    # Greedy Best-First Search
    # ---------------------------------------------------------
    def heuristic(self, a, b):
        a = to_tuple(a)
        b = to_tuple(b)
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def greedy(self, start, goal):
        start = to_tuple(start)
        goal = to_tuple(goal)

        pq = [(self.heuristic(start, goal), start)]
        came_from = {start: None}
        visited = set([start])

        while pq:
            _, (r, c) = heapq.heappop(pq)

            if (r, c) == goal:
                return self.reconstruct(came_from, start, goal)

            for dr, dc in self.DIRS:
                nr, nc = r + dr, c + dc
                if self.in_bounds(nr, nc) and self.is_free(nr, nc):
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        came_from[(nr, nc)] = (r, c)
                        heapq.heappush(pq, (self.heuristic((nr, nc), goal), (nr, nc)))

        return []

    # ---------------------------------------------------------
    # A*
    # ---------------------------------------------------------
    def astar(self, start, goal):
        start = to_tuple(start)
        goal = to_tuple(goal)

        pq = [(0, start)]
        came_from = {start: None}
        g = {start: 0}

        while pq:
            _, (r, c) = heapq.heappop(pq)

            if (r, c) == goal:
                return self.reconstruct(came_from, start, goal)

            for dr, dc in self.DIRS:
                nr, nc = r + dr, c + dc
                if self.in_bounds(nr, nc) and self.is_free(nr, nc):
                    new_g = g[(r, c)] + 1
                    if (nr, nc) not in g or new_g < g[(nr, nc)]:
                        g[(nr, nc)] = new_g
                        f = new_g + self.heuristic((nr, nc), goal)
                        came_from[(nr, nc)] = (r, c)
                        heapq.heappush(pq, (f, (nr, nc)))

        return []

    # ---------------------------------------------------------
    # Wavefront / Brushfire
    # ---------------------------------------------------------
    def wavefront(self, goal):
        goal = to_tuple(goal)

        dist = np.full((self.H, self.W), np.inf)
        queue = deque([goal])
        dist[goal] = 0

        while queue:
            r, c = queue.popleft()
            base = dist[r, c]

            for dr, dc in self.DIRS:
                nr, nc = r + dr, c + dc
                if self.in_bounds(nr, nc) and self.is_free(nr, nc):
                    if dist[nr, nc] > base + 1:
                        dist[nr, nc] = base + 1
                        queue.append((nr, nc))

        return dist

    def wavefront_path(self, dist, start, goal):
        start = to_tuple(start)
        goal = to_tuple(goal)

        if np.isinf(dist[start]):
            return []

        path = [start]
        cur = start

        while cur != goal:
            r, c = cur
            base = dist[r, c]

            best = None
            best_val = base

            for dr, dc in self.DIRS:
                nr, nc = r + dr, c + dc
                if self.in_bounds(nr, nc):
                    val = dist[nr, nc]
                    if val < best_val:
                        best_val = val
                        best = (nr, nc)

            if best is None:
                return []

            cur = best
            path.append(cur)

        return path

    # ---------------------------------------------------------
    # Pasos mínimos
    # ---------------------------------------------------------
    def min_steps(self, start, goal, method="astar"):
        algo = {
            "bfs": self.bfs,
            "dfs": self.dfs,
            "dijkstra": self.dijkstra,
            "greedy": self.greedy,
            "astar": self.astar,
            "wavefront": lambda s, g: self.wavefront_path(self.wavefront(g), s, g)
        }.get(method)

        if algo is None:
            raise ValueError(f"Unknown method: {method}")

        path = algo(start, goal)
        return len(path) - 1 if path else np.inf

import torch

class TorchPathPlanner:
    DIRS = torch.tensor([[1,0], [-1,0], [0,1], [0,-1]], dtype=torch.long)

    def __init__(self, grid_np):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.grid = torch.tensor(grid_np, dtype=torch.int8, device=self.device)
        self.H, self.W = self.grid.shape

    def in_bounds(self, r, c):
        return 0 <= r < self.H and 0 <= c < self.W

    def is_free(self, r, c):
        return self.grid[r, c].item() == 0

    def reconstruct(self, came_from, start, goal):
        if goal not in came_from:
            return []
        path = [goal]
        cur = goal
        while cur != start:
            cur = came_from[cur]
            path.append(cur)
        return path[::-1]

    # ---------------------------------------------------------
    # BFS
    # ---------------------------------------------------------
    def bfs(self, start, goal):
        start = to_tuple(start)
        goal = to_tuple(goal)

        queue = deque([start])
        came_from = {start: None}

        while queue:
            r, c = queue.popleft()

            if (r, c) == goal:
                return self.reconstruct(came_from, start, goal)

            for dr, dc in self.DIRS.cpu().numpy():
                nr, nc = r + int(dr), c + int(dc)
                if self.in_bounds(nr, nc) and self.is_free(nr, nc):
                    if (nr, nc) not in came_from:
                        came_from[(nr, nc)] = (r, c)
                        queue.append((nr, nc))

        return []

    # ---------------------------------------------------------
    # A*
    # ---------------------------------------------------------
    def heuristic(self, a, b):
        a = torch.tensor(a, device=self.device)
        b = torch.tensor(b, device=self.device)
        return torch.sum(torch.abs(a - b)).item()

    def astar(self, start, goal):
        start = to_tuple(start)
        goal = to_tuple(goal)

        pq = [(0, start)]
        came_from = {start: None}
        g = {start: 0}

        while pq:
            _, (r, c) = heapq.heappop(pq)

            if (r, c) == goal:
                return self.reconstruct(came_from, start, goal)

            for dr, dc in self.DIRS.cpu().numpy():
                nr, nc = r + int(dr), c + int(dc)
                if self.in_bounds(nr, nc) and self.is_free(nr, nc):
                    new_g = g[(r, c)] + 1
                    if (nr, nc) not in g or new_g < g[(nr, nc)]:
                        g[(nr, nc)] = new_g
                        f = new_g + self.heuristic((nr, nc), goal)
                        came_from[(nr, nc)] = (r, c)
                        heapq.heappush(pq, (f, (nr, nc)))

        return []

    # ---------------------------------------------------------
    # Wavefront (GPU)
    # ---------------------------------------------------------
    def wavefront(self, goal):
        goal = to_tuple(goal)

        dist = torch.full((self.H, self.W), float("inf"), device=self.device)
        queue = deque([goal])
        dist[goal] = 0

        while queue:
            r, c = queue.popleft()
            base = dist[r, c].item()

            nbrs = self.DIRS + torch.tensor([r, c], device=self.device)

            for nbr in nbrs:
                nr, nc = to_tuple(nbr)
                if self.in_bounds(nr, nc) and self.is_free(nr, nc):
                    if dist[nr, nc].item() > base + 1:
                        dist[nr, nc] = base + 1
                        queue.append((nr, nc))

        return dist

    def wavefront_path(self, dist, start, goal):
        start = to_tuple(start)
        goal = to_tuple(goal)

        if torch.isinf(dist[start]).item():
            return []

        path = [start]
        cur = start

        while cur != goal:
            r, c = cur
            base = dist[r, c].item()

            nbrs = self.DIRS + torch.tensor([r, c], device=self.device)
            best = None
            best_val = base

            for nbr in nbrs:
                nr, nc = to_tuple(nbr)
                if self.in_bounds(nr, nc):
                    val = dist[nr, nc].item()
                    if val < best_val:
                        best_val = val
                        best = (nr, nc)

            if best is None:
                return []

            cur = best
            path.append(cur)

        return path

    # ---------------------------------------------------------
    # Pasos mínimos
    # ---------------------------------------------------------
    def min_steps(self, start, goal, method="astar"):
        algo = {
            "bfs": self.bfs,
            "astar": self.astar,
            "wavefront": lambda s, g: self.wavefront_path(self.wavefront(g), s, g)
        }.get(method)

        if algo is None:
            raise ValueError(f"Unknown method: {method}")

        path = algo(start, goal)
        return len(path) - 1 if path else float("inf")
