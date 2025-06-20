import networkx as nx


def syn_dep_adj_generation(head, dep, vocab_dep):
    syn_dep_edge = []
    for node_s_id, (node_e_id, d) in enumerate(zip(head, dep)):
        if node_e_id == 0:
            continue
        syn_dep_edge.append(
            [node_s_id, node_e_id-1, vocab_dep.stoi.get(d, vocab_dep.unk_index)])
    return syn_dep_edge


def short_adj_generation(head, max_tree_dis=5):
    r'''
    generate short adj matrix
    '''
    head = list(head)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(head)))
    graph.add_edges_from([(node_1, node_2 - 1)
                          for node_1, node_2 in enumerate(head) if node_2 != 0])
    short_adj = [[max_tree_dis]*len(head) for _ in range(len(head))]
    for node_s_id in graph.nodes:
        for node_e_id in graph.nodes:
            try:
                tree_distance = nx.dijkstra_path_length(
                    graph, source=node_s_id, target=node_e_id)
                tree_distance = tree_distance if tree_distance <= max_tree_dis else max_tree_dis
            except:
                tree_distance = max_tree_dis
            short_adj[node_s_id][node_e_id] = tree_distance
    return short_adj
