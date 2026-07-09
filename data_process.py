import torch
import sys
import os
import networkx as nx
from config import get_args
from utils import normalize_features, initialize_feature, deduplicate_communities_list
import numpy as np
import matplotlib.pyplot as plt
import random

def write_graph_pyg(data_list, root):
    adj = data_list[0]
    coalesced_tensor = adj.coalesce()
    index = coalesced_tensor.indices()
    written_edges = set()
    edge_file = open(root + "/graph", "w")
    for i in range(index.shape[1]):
        node1 = index[0][i].item()
        node2 = index[1][i].item()
        if node1 != node2:
            if node1 > node2:
                node1, node2 = node2, node1
            edge = (node1, node2)
            if edge not in written_edges:
                edge_file.write(f"{node1} {node2}\n")
                written_edges.add(edge)
    edge_file.close()

def write_graph_snap(root):
    raw_edge_file = root + "/raw_graph.txt"
    written_edges = set()
    edge_file = open(root + "/graph", "w")
    node_mapping = {}
    current_id = 0
    with open(raw_edge_file, 'r') as r_file:
        for line in r_file:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) == 2:
                node1, node2 = int(parts[0]), int(parts[1])
                if node1 == node2:
                    continue
                if node1 > node2:
                    node1, node2 = node2, node1
                if node1 not in node_mapping:
                    node_mapping[node1] = current_id
                    current_id += 1
                if node2 not in node_mapping:
                    node_mapping[node2] = current_id
                    current_id += 1
                mapped_edge = (node_mapping[node1], node_mapping[node2])
                if mapped_edge not in written_edges:
                    edge_file.write(f"{mapped_edge[0]} {mapped_edge[1]}\n")
                    written_edges.add(mapped_edge)
    edge_file.close()
    return node_mapping

def write_query_gt(query_path, gt_path, selected_queries, ground_truth):
    query_file=open(query_path, "w")
    gt_file = open(gt_path, "w")
    for i in range(len(selected_queries)):
        for j in range(len(selected_queries[i])):
            query_file.write(str(selected_queries[i][j]))
            query_file.write(" ")
        query_file.write("\n")
        for j in range(len(ground_truth[i])):
            gt_file.write(str(ground_truth[i][j]))
            gt_file.write(" ")
        gt_file.write("\n")

def sample_QGT(sample_num, query_num, communities, mask = None):
    queries = []
    gt = []
    query_set = set()
    if query_num > 1: # vertices in a query are from the same community
        i = 0
        while i < sample_num:
            selected_class = torch.randint(0, len(communities), (1,)).item()
            if len(communities[selected_class]) < query_num:
                continue
            sorted_community = sorted(communities[selected_class])
            selected_nodes = random.sample(sorted_community, query_num)
            query_set.update(set(selected_nodes))
            queries.append(selected_nodes)
            gt.append(communities[selected_class])
            i = i + 1
    elif query_num == 1:
        if mask is None:
            mask = set()
        community_available_nodes = []
        for community in communities:
            available = sorted([node for node in community if node not in mask])
            community_available_nodes.append(set(available))
        
        i = 0
        actual_sample_num = 0
        allow_reuse = False
        while i < sample_num:
            available_communities = [idx for idx in range(len(communities)) 
                                    if len(community_available_nodes[idx]) > 0]
            
            if len(available_communities) == 0:
                if not allow_reuse:
                    # print(f"Warning: No available nodes after excluding mask. "
                    #       f"Allowing reuse of masked nodes and previously selected nodes to generate samples.")
                    allow_reuse = True
            
            if allow_reuse:
                all_available_communities = [idx for idx in range(len(communities)) 
                                            if len(communities[idx]) > 0]
                if len(all_available_communities) == 0:
                    raise ValueError(f"sample_num ({sample_num}) exceeds the number of available nodes (0). "
                                   f"No nodes available in the given communities.")
                selected_class = random.choice(all_available_communities)
                selected_node = random.choice(sorted(communities[selected_class]))
            else:
                selected_class = random.choice(available_communities)
                available_nodes_list = sorted(list(community_available_nodes[selected_class]))
                selected_node = random.choice(available_nodes_list)
                community_available_nodes[selected_class].remove(selected_node)
            
            query_set.add(selected_node)
            queries.append([selected_node])
            gt.append(communities[selected_class])
            
            i += 1
            actual_sample_num += 1
    return queries, gt, query_set

def generate_induct_QGT(maxQueryNum, args, root, communities):
    num_class = len(communities)
    # split the groundtruth with 1:1 ratio
    train_size = int(num_class / 2)
    random_coms = torch.randperm(num_class)
    train_index = random_coms[:train_size].tolist()
    test_index = random_coms[train_size:].tolist()
    train_communities = [communities[i] for i in train_index]
    test_communities = [communities[i] for i in test_index]

    total_tv_samples = args.train_sample + args.valid_sample
    all_tv_queries, all_tv_gt, _ = sample_QGT(total_tv_samples, maxQueryNum, train_communities, None)

    indices = list(range(total_tv_samples))
    random.shuffle(indices)
    train_idx = indices[:args.train_sample]
    valid_idx = indices[args.train_sample:]

    train_queries = [all_tv_queries[i] for i in train_idx]
    train_gt = [all_tv_gt[i] for i in train_idx]
    valid_queries = [all_tv_queries[i] for i in valid_idx]
    valid_gt = [all_tv_gt[i] for i in valid_idx]

    test_queries, test_gt, _ = sample_QGT(args.test_sample, maxQueryNum, test_communities, None)

    train_query_file = root + f"/{maxQueryNum}_induct_train_query"
    train_gt_file = root + f"/{maxQueryNum}_induct_train_gt"
    valid_query_file = root + f"/{maxQueryNum}_induct_valid_query"
    valid_gt_file = root + f"/{maxQueryNum}_induct_valid_gt"
    test_query_file = root + f"/{maxQueryNum}_induct_test_query"
    test_gt_file = root + f"/{maxQueryNum}_induct_test_gt"
    write_query_gt(train_query_file, train_gt_file, train_queries, train_gt)
    write_query_gt(valid_query_file, valid_gt_file, valid_queries, valid_gt)
    write_query_gt(test_query_file, test_gt_file, test_queries, test_gt)

def generate_nested_query_QGT(args, root, communities):
    num_class = len(communities)
    train_size = int(num_class / 2)
    random_coms = torch.randperm(num_class)
    train_index = random_coms[:train_size].tolist()
    test_index = random_coms[train_size:].tolist()
    train_communities = [communities[i] for i in train_index]
    test_communities = [communities[i] for i in test_index]
    
    total_tv_samples = args.train_sample + args.valid_sample
    all_tv_queries_8, all_tv_gt_8, _ = sample_QGT(total_tv_samples, 8, train_communities, None)

    indices = list(range(total_tv_samples))
    random.shuffle(indices)
    train_idx = indices[:args.train_sample]
    valid_idx = indices[args.train_sample:]

    train_queries_8 = [all_tv_queries_8[i] for i in train_idx]
    train_gt_8 = [all_tv_gt_8[i] for i in train_idx]
    valid_queries_8 = [all_tv_queries_8[i] for i in valid_idx]
    valid_gt_8 = [all_tv_gt_8[i] for i in valid_idx]

    test_queries_8, test_gt_8, _ = sample_QGT(args.test_sample, 8, test_communities, None)
    
    train_queries_6 = [q[:6] for q in train_queries_8]
    valid_queries_6 = [q[:6] for q in valid_queries_8]
    test_queries_6 = [q[:6] for q in test_queries_8]
    
    train_queries_4 = [q[:4] for q in train_queries_8]
    valid_queries_4 = [q[:4] for q in valid_queries_8]
    test_queries_4 = [q[:4] for q in test_queries_8]
    
    train_queries_2 = [q[:2] for q in train_queries_8]
    valid_queries_2 = [q[:2] for q in valid_queries_8]
    test_queries_2 = [q[:2] for q in test_queries_8]
    
    train_queries_1 = [q[:1] for q in train_queries_8]
    valid_queries_1 = [q[:1] for q in valid_queries_8]
    test_queries_1 = [q[:1] for q in test_queries_8]
    
    query_sizes = [1, 2, 4, 6, 8]
    all_train_queries = [train_queries_1, train_queries_2, train_queries_4, train_queries_6, train_queries_8]
    all_valid_queries = [valid_queries_1, valid_queries_2, valid_queries_4, valid_queries_6, valid_queries_8]
    all_test_queries = [test_queries_1, test_queries_2, test_queries_4, test_queries_6, test_queries_8]
    all_train_gt = [train_gt_8] * 5
    all_valid_gt = [valid_gt_8] * 5
    all_test_gt = [test_gt_8] * 5
    
    for size, train_q, valid_q, test_q, train_g, valid_g, test_g in zip(
        query_sizes, all_train_queries, all_valid_queries, all_test_queries,
        all_train_gt, all_valid_gt, all_test_gt
    ):
        train_query_file = root + f"/{size}_qsize_train_query"
        train_gt_file = root + f"/{size}_qsize_train_gt"
        valid_query_file = root + f"/{size}_qsize_valid_query"
        valid_gt_file = root + f"/{size}_qsize_valid_gt"
        test_query_file = root + f"/{size}_qsize_test_query"
        test_gt_file = root + f"/{size}_qsize_test_gt"
        write_query_gt(train_query_file, train_gt_file, train_q, train_g)
        write_query_gt(valid_query_file, valid_gt_file, valid_q, valid_g)
        write_query_gt(test_query_file, test_gt_file, test_q, test_g)

def generate_nested_query_from_file(args, root):
    output_path = root + "/community"
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Community file not found: {output_path}. "
                               f"Please generate community file first.")
    communities = []
    with open(output_path, 'r') as file:
        for line in file:
            community = list(map(int, line.strip().split()))
            if community:
                communities.append(community)
    
    generate_nested_query_QGT(args, root, communities)


def read_communities_from_file(root):
    output_path = root + "/community"
    if not os.path.exists(output_path):
        raise FileNotFoundError(
            f"Community file not found: {output_path}. "
            f"Please generate community file first (e.g., via process_data)."
        )
    communities = []
    with open(output_path, 'r') as file:
        for line in file:
            community = list(map(int, line.strip().split()))
            if community:
                communities.append(community)
    return communities


def regenerate_pyg_queries_from_existing(args, dataset_names=None):
    datasets_to_process = dataset_names if dataset_names is not None else args.dataset_pyg
    
    for name in datasets_to_process:
        if name not in args.dataset_pyg:
            continue
        root = os.path.join(args.data_path, name)

        communities = read_communities_from_file(root)

        maxQueryNum = 1
        generate_induct_QGT(maxQueryNum, args, root, communities)
        generate_nested_query_QGT(args, root, communities)

        graphx = nx.Graph()
        graph_path = root + "/graph"
        with open(graph_path, "r") as edge_file:
            for line in edge_file:
                node1, node2 = map(int, line.split())
                if node1 != node2:
                    graphx.add_edge(node1, node2)

        generate_boundary_test_QGT(graphx, maxQueryNum, args, root)


def regenerate_snap_queries_from_existing(args, dataset_names=None):
    datasets_to_process = dataset_names if dataset_names is not None else args.dataset_snap
    
    for name in datasets_to_process:
        if name not in args.dataset_snap:
            continue
        root = os.path.join(args.data_path, name)

        communities = read_communities_from_file(root)

        maxQueryNum = 1
        generate_induct_QGT(maxQueryNum, args, root, communities)
        generate_nested_query_QGT(args, root, communities)

        graphx = nx.Graph()
        graph_path = root + "/graph"
        with open(graph_path, "r") as edge_file:
            for line in edge_file:
                node1, node2 = map(int, line.split())
                if node1 != node2:
                    graphx.add_edge(node1, node2)

        generate_boundary_test_QGT(graphx, maxQueryNum, args, root)

def get_boundary_nodes(G, community):
    community_set = set(community)
    boundary_nodes = []
    
    for node in community:
        if node not in G:
            continue
        neighbors = set(G.neighbors(node))
        if neighbors - community_set:
            boundary_nodes.append(node)
    
    return boundary_nodes

def generate_boundary_test_QGT(G, maxQueryNum, args, root):
    induct_gt_file = root + f"/{maxQueryNum}_induct_test_gt"
    
    if not os.path.exists(induct_gt_file):
        raise FileNotFoundError(f"Inductive test gt file not found: {induct_gt_file}. "
                               f"Please generate inductive queries (mod=1) first.")
    
    test_communities = []
    with open(induct_gt_file, 'r') as f:
        for line in f:
            gt = list(map(int, line.strip().split()))
            test_communities.append(gt)
    
    test_communities = deduplicate_communities_list(test_communities)
    test_communities = [list(com) for com in test_communities]
    
    communities_with_boundary = []
    for community in test_communities:
        boundary_nodes = get_boundary_nodes(G, community)
        if len(boundary_nodes) > 0:
            community_set = set(community)
            boundary_set = set(boundary_nodes)
            non_boundary_nodes = sorted(list(community_set - boundary_set))
            if len(non_boundary_nodes) > 0:
                communities_with_boundary.append({
                    'community': community,
                    'boundary': boundary_nodes,
                    'non_boundary': non_boundary_nodes
                })
    
    if len(communities_with_boundary) == 0:
        print(f"Warning: No test communities with both boundary and non-boundary nodes found.")
        boundary_query_file = root + f"/{maxQueryNum}_boundary_test_query"
        boundary_gt_file = root + f"/{maxQueryNum}_boundary_test_gt"
        noboundary_query_file = root + f"/{maxQueryNum}_noboundary_test_query"
        noboundary_gt_file = root + f"/{maxQueryNum}_noboundary_test_gt"
        write_query_gt(boundary_query_file, boundary_gt_file, [], [])
        write_query_gt(noboundary_query_file, noboundary_gt_file, [], [])
        return
    
    selected_boundary_nodes = set()
    selected_noboundary_nodes = set()
    
    boundary_queries = []
    boundary_gt = []
    noboundary_queries = []
    noboundary_gt = []
    
    while True:
        if len(boundary_queries) > args.test_sample:
            break
        
        available_communities = []
        for com_info in communities_with_boundary:
            community = com_info['community']
            boundary_nodes = com_info['boundary']
            non_boundary_nodes = com_info['non_boundary']
            
            available_boundary = sorted([node for node in boundary_nodes if node not in selected_boundary_nodes])
            available_noboundary = sorted([node for node in non_boundary_nodes if node not in selected_noboundary_nodes])
            
            if len(available_boundary) > 0 and len(available_noboundary) > 0:
                available_communities.append({
                    'community': community,
                    'available_boundary': available_boundary,
                    'available_noboundary': available_noboundary
                })
        
        if len(available_communities) == 0:
            break
        
        selected_com_info = random.choice(available_communities)
        community = selected_com_info['community']
        available_boundary = selected_com_info['available_boundary']
        available_noboundary = selected_com_info['available_noboundary']
        
        sorted_available_boundary = sorted(available_boundary)
        selected_boundary = random.choice(sorted_available_boundary)
        
        sorted_available_noboundary = sorted(available_noboundary)
        selected_noboundary = random.choice(sorted_available_noboundary)
        
        selected_boundary_nodes.add(selected_boundary)
        selected_noboundary_nodes.add(selected_noboundary)
        
        boundary_queries.append([selected_boundary])
        boundary_gt.append(community)
        noboundary_queries.append([selected_noboundary])
        noboundary_gt.append(community)
    
    # write query and gt
    boundary_query_file = root + f"/{maxQueryNum}_boundary_test_query"
    boundary_gt_file = root + f"/{maxQueryNum}_boundary_test_gt"
    noboundary_query_file = root + f"/{maxQueryNum}_noboundary_test_query"
    noboundary_gt_file = root + f"/{maxQueryNum}_noboundary_test_gt"
    write_query_gt(boundary_query_file, boundary_gt_file, boundary_queries, boundary_gt)
    write_query_gt(noboundary_query_file, noboundary_gt_file, noboundary_queries, noboundary_gt)

def process_communities(G, communities, com_limit):
    if not communities:
        return []
    
    connected_communities = []
    for community in communities:
        if not community:
            continue
        subgraph = G.subgraph(community)
        if len(subgraph) == 0:
            continue
        connected_components = sorted(nx.connected_components(subgraph), key=lambda x: (len(x), min(x)))
        for cc in connected_components:
            if len(cc) >= com_limit:
                connected_communities.append(sorted(list(cc)))
    
    if not connected_communities:
        return []
    
    connected_communities.sort(key=len, reverse=True)
    
    node_to_community = {}
    final_communities = []
    
    for community in connected_communities:
        overlapping_nodes = [node for node in community if node in node_to_community]
        
        if overlapping_nodes:
            non_overlapping = [node for node in community if node not in node_to_community]
            if non_overlapping:
                subgraph = G.subgraph(non_overlapping)
                if len(subgraph) > 0:
                    connected_components = sorted(nx.connected_components(subgraph), key=lambda x: (len(x), min(x)))
                    for cc in connected_components:
                        if len(cc) >= com_limit:
                            sorted_cc = sorted(list(cc))
                            final_communities.append(sorted_cc)
                            for node in sorted_cc:
                                node_to_community[node] = len(final_communities) - 1
        else:
            sorted_community = sorted(community)
            final_communities.append(sorted_community)
            for node in sorted_community:
                node_to_community[node] = len(final_communities) - 1
    
    final_communities = [com for com in final_communities if len(com) >= com_limit]
    
    return final_communities

def generate_query_pyg(G, data_list, root, args, mod=1):
    labels = data_list[2]
    num_class = (torch.max(labels)-torch.min(labels)+1).item()

    communities = []
    for j in range(num_class):
        nodes_in_class = [i for i in range(labels.shape[0]) if labels[i] == j]
        if nodes_in_class:
            communities.append(nodes_in_class)
    
    communities = process_communities(G, communities, args.com_limit)

    # wirte communities
    output_path = root + "/community"
    with open(output_path, 'w') as file:
        for community in communities:
            line = ' '.join(map(str, community))
            file.write(line + '\n')

    # STEP2: generate query and groundtruth

    if mod == 1: # inductive
        maxQueryNum = 1
        generate_induct_QGT(maxQueryNum, args, root, communities)

    elif mod == 3: # generate nested queries with size 1,2,4,6,8
        generate_nested_query_QGT(args, root, communities)
    
    elif mod == 9 or mod == 10: # generate boundary test query
        maxQueryNum = 1
        generate_boundary_test_QGT(G, maxQueryNum, args, root)

def generate_query_snap(G, node_mapping, root, args, mod=1, isEmail = False):
    file_path = root + "/raw_community.txt"
    communities = []
    if not isEmail:
        with open(file_path, 'r') as file:
            for line in file:
                community = list(map(int, line.strip().split()))
                filtered_community = [node_mapping[node] for node in community if node in node_mapping]
                if filtered_community:
                    communities.append(filtered_community)
    else:
        class_to_nodes = {}
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 2:
                    node_id = int(parts[0])
                    class_id = int(parts[1])
                    if class_id not in class_to_nodes:
                        class_to_nodes[class_id] = []
                    class_to_nodes[class_id].append(node_id)
        
        for class_id, community in class_to_nodes.items():
            filtered_community = [node_mapping[node] for node in community if node in node_mapping]
            if filtered_community:
                communities.append(filtered_community)
    
    communities = process_communities(G, communities, args.com_limit)

    # wirte communities
    output_path = root + "/community"
    with open(output_path, 'w') as file:
        for community in communities:
            line = ' '.join(map(str, community))
            file.write(line + '\n')

    # STEP2: generate query and groundtruth
    if mod == 1: # inductive
        maxQueryNum = 1
        generate_induct_QGT(maxQueryNum, args, root, communities)

    elif mod == 3: # generate nested queries with size 1,2,4,6,8
        generate_nested_query_QGT(args, root, communities)
    
    elif mod == 9 or mod == 10: # generate boundary test query
        maxQueryNum = 1
        generate_boundary_test_QGT(G, maxQueryNum, args, root)

def process_data(args, type):    
    # process data from pyg
    if type == "pyg":
        for name in args.dataset_pyg:
            root = os.path.join(args.data_path, name)
            # load raw data
            data_list = torch.load(os.path.join(root, "data.pt"))
            feats = data_list[1]
            num_node = feats.shape[0]

            # write graph
            write_graph_pyg(data_list, root)

            # compute structure features (e.g. normalized core number)
            graphx = nx.Graph()
            graphPath = root + "/graph"
            with open(graphPath, "r") as edge_file:
                for line in edge_file:
                    node1, node2 = map(int, line.split())
                    if node1 != node2:
                        graphx.add_edge(node1, node2)
            print(f"data: {root} nodes: {graphx.number_of_nodes()} / {num_node} edges: {graphx.number_of_edges()}")
            initialize_feature(graphx, num_node, root, 1)

            # initialize attribute feature
            feats = normalize_features(data_list[1].numpy())
            attr_file = root + "/feat_attr.npy"
            np.save(attr_file, feats)

            # generate query
            for mod in [1,3,9]:
                generate_query_pyg(graphx, data_list, root, args, mod)

    # process data from snap
    elif type == "snap":
        for name in args.dataset_snap:
            root = os.path.join(args.data_path, name)
            # write graph
            node_mapping = write_graph_snap(root)

            # compute structure features (e.g. normalized core number)
            graphx = nx.Graph()
            graphPath = root + "/graph"
            with open(graphPath, "r") as edge_file:
                for line in edge_file:
                    node1, node2 = map(int, line.split())
                    if node1 != node2:
                        graphx.add_edge(node1, node2)
            max_id = max(graphx.nodes) + 1 if graphx.nodes else 0
            print(f"data: {root} nodes: {graphx.number_of_nodes()} / {max_id} edges: {graphx.number_of_edges()}")
            initialize_feature(graphx, max_id, root, 1)

            # generate query
            for mod in [1,3,9]:
                if name == "email":
                    isEmail = True
                else:
                    isEmail = False
                generate_query_snap(graphx, node_mapping, root, args, mod, isEmail)

# mod==0 sample different percentage vertices, mod==1 insert and delete different percentage edges
def generateGraph(args, mod):
    all_data = args.dataset_pyg + args.dataset_snap
    for name in all_data:
        print(f"dataset name: {name}")
        data_file = os.path.join(args.data_path, name)

        # read feat
        feature = []
        feat_file = data_file + "/feat"
        with open(feat_file, 'r') as file:
            feature = np.array([[float(line.split()[1])] for line in file],
                               dtype=np.float32)
        N = feature.shape[0]

        # read origin graph
        graph_file = data_file + "/graph"
        G = nx.Graph()
        G.add_nodes_from(range(N))
        with open(graph_file, 'r') as file:
            for line in file:
                node1, node2 = map(int, line.split())
                G.add_edge(node1, node2)
        print(f"number of nodes: {N}")
        print(f"number of nodes with neighbors: {G.number_of_nodes()}")
        print(f"number of edges: {G.number_of_edges()}")

        if mod == 0:
            # sample 20%, 40%, 60%, 80% vertices
            node_ids = list(range(N))
            random.shuffle(node_ids)

            for ratio in [0.2, 0.4, 0.6, 0.8]:
                k = int(N * ratio)
                sampled_nodes = sorted(node_ids[:k])

                filename = data_file + "/" + f"{int(ratio * 100)}_vertex"
                with open(filename, "w") as f:
                    for node in sampled_nodes:
                        f.write(f"{node}\n")
        elif mod == 1:
            # insert and delete 10%, 20%, 30% edges
            all_nodes = list(G.nodes())
            existing_edges = set(tuple(sorted(edge)) for edge in G.edges())
            num_edges = len(existing_edges)

            def sample_non_existing_edges(k):
                sampled = set()
                attempts = 0
                while len(sampled) < k:
                    u, v = random.sample(all_nodes, 2)
                    edge = tuple(sorted((u, v)))
                    if edge in existing_edges or edge in sampled:
                        continue
                    sampled.add(edge)
                    attempts += 1
                return list(sampled)

            for ratio in [0.1, 0.2, 0.3]:
                k = int(num_edges * ratio)
                # delete edges
                del_edges = random.sample(list(existing_edges), k)
                with open(data_file + "/" + f"del_{int(ratio * 100)}_edge", "w") as f:
                    for u, v in del_edges:
                        f.write(f"{u} {v}\n")
                # add edges
                add_edges = sample_non_existing_edges(k)
                with open(data_file + "/" + f"add_{int(ratio * 100)}_edge", "w") as f:
                    for u, v in add_edges:
                        f.write(f"{u} {v}\n")

if __name__ == "__main__":
    args = get_args()
    print("\nParsed Arguments:")
    print(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    process_data(args, "pyg")
    process_data(args, "snap")

    generateGraph(args, 0)
    generateGraph(args, 1) 