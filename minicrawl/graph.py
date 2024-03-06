


class Graph:
    def __init__(self, directed):
        self._graph = {}
        self._directed = directed

    def add_node(self, node):
        self._graph[node] = set()

    def add_egde(self, node1, node2):
        self._graph[node1].add(node2)
        if not self._directed:
            self._graph[node2].add(node1)

    def remove_node(self, node):
        self._graph.pop(node)
        for k in self._graph.keys():
            if node in self._graph[k]:
                self._graph[k].remove(node)

    def remove_edge(self, node1, node2):
        self._graph[node1].remove(node2)
        if not self._directed:
            self._graph[node2].remove(node1)

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

    def dfs(self, node, discovered=[], depth=0):
        if node not in discovered:
            discovered.append(node)
            depth += 1
            if node in self._graph.keys():
                for edge in self._graph[node]:
                    discovered, depth = self.dfs(edge, discovered, depth)

        return discovered, depth

    def connected_components(self, components=[], discovered=[]):
        for node in self._graph.keys():
            if node not in discovered:
                discovered.append(node)
                for edge in self._graph[node]:
                    if edge not in discovered:
                        discovered.append(edge)
            components.append(discovered)

        return components


    def get_nodes(self):
        return list(self._graph.keys())

    def get_edges(self, node):
        return list(self._graph[node])


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
