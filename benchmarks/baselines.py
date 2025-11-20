import heapq


class StandardBaselines:
    """
    Standard Pathfinding Algorithms (A*, Dijkstra) for comparison.
    """

    @staticmethod
    def run_dijkstra(G, start, goal):
        visited = set()
        pq = [(0, start)]
        costs = {start: 0}

        while pq:
            cost, u = heapq.heappop(pq)
            if u in visited: continue
            visited.add(u)

            if u == goal:
                return {'status': 'success', 'path_cost': cost}

            for v in G.neighbors(u):
                weight = G[u][v].get('weight', 1.0)
                new_cost = cost + weight
                if new_cost < costs.get(v, float('inf')):
                    costs[v] = new_cost
                    heapq.heappush(pq, (new_cost, v))
        return {'status': 'failure'}

    @staticmethod
    def run_astar(G, start, goal):
        def heuristic(a, b):
            # Manhattan distance
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        visited = set()
        pq = [(0, start)]  # Priority Queue: (f_score, node)
        g_score = {start: 0}

        while pq:
            _, u = heapq.heappop(pq)

            if u in visited: continue
            visited.add(u)

            if u == goal:
                return {'status': 'success', 'path_cost': g_score[u]}

            for v in G.neighbors(u):
                weight = G[u][v].get('weight', 1.0)
                tentative_g = g_score[u] + weight

                if tentative_g < g_score.get(v, float('inf')):
                    g_score[v] = tentative_g
                    f_score = tentative_g + heuristic(v, goal)
                    heapq.heappush(pq, (f_score, v))

        return {'status': 'failure'}