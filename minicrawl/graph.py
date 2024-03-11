from copy import deepcopy


class Graph:
    def __init__(self, directed):
        self._graph = {}
        self._directed = directed

    def add_node(self, node):
        self._graph[node] = set()

    def add_egde(self, node1, node2):
        self._graph[node1].add(node2)
        if not self._directed:
            if node2 not in self._graph.keys():
                self._graph[node2] = set()
            self._graph[node2].add(node1)

    def remove_node(self, node):
        self._graph.pop(node)
        for k in self._graph.keys():
            if node in self._graph[k]:
                self._graph[k].remove(node)

    def remove_edge(self, node1, node2):
        try:
            self._graph[node1].remove(node2)
            if not self._directed:
                self._graph[node2].remove(node1)
        except KeyError:
            print("Already removed.")

    def find_path(self, node1, node2, path=[]):
        path.append(node1)
        if node1 == node2:
            return path
        for n in self._graph[node1]:
            if self._graph[node1] is not None and n not in path:
                path = self.find_path(n, node2, path)
                if path:
                    return path
                else:
                    return None

        return None

    def longest_walkable_path(self):
        max_distance = 0
        for node in self._graph.keys():
            distance = self._walk_to_leaf(node, distance=0)
            if distance > max_distance:
                max_distance = distance

        return max_distance

    def _walk_to_leaf(self, node, distance):
        if node in list(self._graph.keys()):
            distance += 1
            for edge in self._graph[node]:
                return self._walk_to_leaf(edge, distance)

        return distance

    def dfs(self, node, discovered, visited):
        visited[node] = True
        discovered.append(node)
        for e in self._graph[node]:
            if visited[e] == False:
                discovered = self.dfs(e, discovered, visited)

        return discovered

    def bfs(self, node):
        visited = {}
        for k in self._graph.keys():
            visited[k] = False

        queue = [node]
        visited[node] = True

        while queue:
            n = queue.pop(0)
            for e in self._graph[n]:
                if not visited[e]:
                    queue.append(e)
                    visited[e] = True

    def minimum_spanning_tree(self, node, replace=False):
        mst = deepcopy(self._graph)
        visited = {}
        for k in self._graph.keys():
            visited[k] = False

        queue = [node]
        visited[node] = True

        while queue:
            n = queue.pop(0)
            for e in self._graph[n]:
                if not visited[e]:
                    queue.append(e)
                    visited[e] = True
                else:
                    mst[n].remove(e)

        if replace:
            self._graph = mst

        return mst

    def connected_components(self):
        # Initialization
        visited = {}
        components = []
        for n in self._graph.keys():
            visited[n] = False
        # Start search
        for n in self._graph.keys():
            if visited[n] == False:
                comp = []
                components.append(self.dfs(n, comp, visited))

        return components

    def get_nodes(self):
        return list(self._graph.keys())

    def get_edges(self, node):
        return list(self._graph[node])

    def connect_components(self, comp1, comp2):
        """
        Try to find an easy connection between comp1 and comp2.
        Check whether comp2's root is a manhattan-neighbor of some comp1 node
        :param comp1: list of nodes. First component (one that will be retained regardless of the outcome)
        :param comp2: list of nodes. Component that we are trying to connect.
        :return: None
        """
        for n in comp2:
            neighbors = self._get_manhattan_neighbors(n)
            found = False
            for neigh in neighbors:
                # If a match is found, add an edge between the two nodes
                if neigh in comp1:
                    # Just in case
                    if neigh not in self._graph.keys():
                        self._graph[neigh] = set()
                    self.add_egde(neigh, n)
                    found = True
                    break
            if found:
                break

        # If no easy connection has been found... "bye bye, smaller component!"
        if not found:
            for e in self._graph[comp2[0]]:
                self._graph.pop(e)
            self._graph.pop(comp2[0])

        self._turn_to_directed()

    @staticmethod
    def _get_manhattan_neighbors(node):
        return [(node[0] - 1, node[1]), (node[0] + 1, node[1]), (node[0], node[1] - 1), (node[0], node[1] + 1)]

    def _turn_to_directed(self):
        visited = []
        for n in self._graph.keys():
            visited.append(n)
            for e in self._graph[n]:
                try:
                    self._graph[e].remove(n)
                except Exception:
                    continue


if __name__ == '__main__':
    g = Graph(directed=True)
    nodes = ['A', 'B', 'C', 'D', 'E', 'F']
    edges = [('A', 'B'), ('B', 'C'), ('B', 'D'), ('C', 'D'), ('E', 'F'), ('F', 'C')]
    for n in nodes:
        g.add_node(n)
    for n1, n2 in edges:
        g.add_egde(n1, n2)

    path = g.find_path('A', 'C')
    print(path)
