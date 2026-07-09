# file.py

from greedy_algorithm import GREEDYALGORITHM

def ReadGraph(filename, my_algorithm: GREEDYALGORITHM):
    # read the graph from file and store it in my_algorithm
    try:
        with open(filename, 'r') as infile:
            n, m = map(int, infile.readline().split())
            my_algorithm.n = n
            my_algorithm.m = m
            my_algorithm.graph = [[] for _ in range(n + 1)]
            for _ in range(m):
                line = infile.readline()
                if not line:
                    break
                u, v = map(int, line.split())
                my_algorithm.graph[u].append(v)
                my_algorithm.graph[v].append(u)
                # print(u, v)
    except Exception as e:
        print(f"Error: cannot open file {filename}")
        return

def ReadQueries(filename, my_algorithm: GREEDYALGORITHM):
    # read the queries from file and store them in my_algorithm
    try:
        with open(filename, 'r') as infile:
            for line in infile:
                if not line.strip():
                    continue
                u = int(line.strip())
                my_algorithm.community_set.add(u)
    except Exception as e:
        print(f"Error: cannot open file {filename}")
        return

def WriteCommunity(filename, my_algorithm: GREEDYALGORITHM):
    # write the community set to file
    try:
        with open(filename, 'w') as outfile:
            for u in my_algorithm.community_set:
                for v in my_algorithm.graph[u]:
                    if v in my_algorithm.community_set and u < v:
                        outfile.write(f"{u} {v}\n")
    except Exception as e:
        print(f"Error: cannot open file {filename}")
        return
