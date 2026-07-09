import heapq

def heap_com(seed, graph, gnnsroces):
    queue = []
    vis = {}
    heapq.heappush(queue, (-gnnsroces[seed], seed))
    vis[seed] = 1
    topk = []
    sum_s = -gnnsroces[seed]
    while queue:  # and len(topk)<subgraphs.max_com:
        sroce, node = heapq.heappop(queue)
        sroce = -sroce
        # print(node,sroce,sum_s/len(topk))
        if len(topk) < 3 or sroce >= sum_s / len(topk):
            sum_s += sroce
            topk.append(node)
            for u in graph.neighbors(node):
                if u not in vis.keys():
                    vis[u] = 1
                    heapq.heappush(queue, (-gnnsroces[u], u))
    return topk


def heap_com_topk(seed, graph, gnnsroces, max_size):
    queue = []
    vis = {}
    heapq.heappush(queue, (-gnnsroces[seed], seed))
    vis[seed] = 1
    topk = []
    sum_s = -gnnsroces[seed]
    while queue and len(topk) < max_size:
        sroce, node = heapq.heappop(queue)
        sroce = -sroce
        sum_s += sroce
        topk.append(node)
        for u in graph.neighbors(node):
            if u not in vis.keys():
                vis[u] = 1
                heapq.heappush(queue, (-gnnsroces[u], u))
    return topk


def calu(coms, valid):
    n_valid = len(valid)
    brf, brj = 0, 0
    i = 0
    for precom in valid:
        # if i and i %500==0:
        #    print(f'valid {i} brf,brj',brf/i,brj/i)
        i += 1
        f, j = 0, 0
        for right_com in coms:
            right = len(set(right_com) & set(precom))
            nr = right / len(right_com)
            np = right / len(precom)
            j = max(j, len(set(right_com) & set(precom)) / len(set(right_com) | set(precom)))
            if nr + np != 0:
                tmpf = 2 * (nr * np) / (nr + np)
                if tmpf > f:
                    f = tmpf
        brf += f
        brj += j
    return brf / n_valid, brj / n_valid


def heap_com_threshold(seed, graph, gnnsroces, threshold):
    queue = []
    vis = {}
    heapq.heappush(queue, (-gnnsroces[seed], seed))
    vis[seed] = 1
    topk = []
    sum_s = 0
    while queue:  # and len(topk)<subgraphs.max_com:
        sroce, node = heapq.heappop(queue)
        sroce = -sroce
        if len(topk) < 3 or sroce >= threshold:
            topk.append(node)
            for u in graph.neighbors(node):
                if u not in vis.keys():
                    vis[u] = 1
                    heapq.heappush(queue, (-gnnsroces[u], u))
    return topk


def heap_com_threshold_nonc(seed, graph, gnnsroces, threshold, nodes, shift):
    topk = []
    for node in nodes:
        if gnnsroces[node + shift] >= threshold:
            topk.append(node + shift)
    if seed not in topk:
        topk.append(seed)
    return topk
