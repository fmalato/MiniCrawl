from copy import deepcopy

from minicrawl.graph import Graph


def minimum_spanning_tree(graph, node):
    mst = Graph(directed=False)
    visited = {}
    for k in graph.get_nodes():
        mst.add_node(k)
        visited[k] = False

    queue = [node]
    visited[node] = True

    while queue:
        n = queue.pop(0)
        for e in graph.get_edges(n):
            if not visited[e]:
                queue.append(e)
                visited[e] = True
                mst.add_egde(n, e)

    return mst
