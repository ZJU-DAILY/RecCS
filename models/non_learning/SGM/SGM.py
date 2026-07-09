import sys
from collections import deque

graph = []
community = set()
neigh_set = set()
modularity = 0.0
n = 0
m = 0

def ReadGraph(path):
    global n, m, graph
    with open(path, 'r') as infile:
        n, m = map(int, infile.readline().split())
        graph = [[] for _ in range(n + 1)]
        for _ in range(m):
            u, v = map(int, infile.readline().split())
            graph[u].append(v)
            graph[v].append(u)

def ReadQueries(path):
    with open(path, 'r') as infile:
        for line in infile:
            for q in line.strip().split():
                if q:
                    community.add(int(q))

def OutputCommunity(path):
    with open(path, 'w') as outfile:
        outfile.write(' '.join(str(node) for node in community) + '\n')

def CalculateModularity():
    ind = 0
    outd = 0
    for node in community:
        for neighbor in graph[node]:
            if neighbor in community:
                if node < neighbor:
                    ind += 1
            else:
                outd += 1
    # print(f'ind: {ind} outd: {outd}')
    if outd == 0:
        return float('inf')
    return ind / outd

def CalculateNeighSet():
    global neigh_set
    neigh_set = set()
    for node in community:
        for neighbor in graph[node]:
            if neighbor not in community:
                degree = len(graph[neighbor])
                neigh_set.add((degree, neighbor))

def CalculateDeltaModularity(node):
    global modularity
    delta_modularity = 0
    if node in community:
        community.remove(node)
        new_modularity = CalculateModularity()
        delta_modularity = new_modularity - modularity
        community.add(node)
    else:
        community.add(node)
        new_modularity = CalculateModularity()
        delta_modularity = new_modularity - modularity
        community.remove(node)
    return delta_modularity

def IsConnectedAfterDelete(node):
    if not community:
        return False
    start = next(iter(community))
    q = deque()
    q.append(start)
    visited = set([start])
    while q:
        cur = q.popleft()
        for neighbor in graph[cur]:
            if neighbor in community and neighbor not in visited and neighbor != node:
                visited.add(neighbor)
                q.append(neighbor)
    return len(visited) == len(community)

def LocalOptimalAlgorithm():
    global modularity
    CalculateNeighSet()
    modularity = CalculateModularity()
    has_added = True
    added_nodes = set()
    while has_added or added_nodes:
        has_added = False
        added_nodes.clear()
        for degree, node in sorted(neigh_set, reverse=True):
            delta_modularity = CalculateDeltaModularity(node)
            if delta_modularity > 0:
                community.add(node)
                modularity = CalculateModularity()
                added_nodes.add(node)
                # print(node, modularity)
        # print("remove:")
        community_copy = set(community)
        for node in community_copy:
            delta_modularity = CalculateDeltaModularity(node)
            if delta_modularity > 0 and IsConnectedAfterDelete(node):
                community.remove(node)
                modularity = CalculateModularity()
                added_nodes.discard(node)
                # print(node, modularity)
        CalculateNeighSet()

def main():
    graph_path = sys.argv[1]
    query_path = sys.argv[2]
    output_path = sys.argv[3]
    ReadGraph(graph_path)
    ReadQueries(query_path)
    LocalOptimalAlgorithm()
    OutputCommunity(output_path)

if __name__ == '__main__':
    main()