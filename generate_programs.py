import graphs

def generate_tree_program_graph(layers=2, branching=3, compute=4, data=1, vflags=None):

    def generate_tree(layers=2, branching=3, compute=4, data=1, vflags=None):
        if layers == 0:
            return None, [], []

        if vflags is None:
            vflags = {"machine": 1, "rack_switch": 0, "root_switch": 0}

        # add coalescing node for this layer
        c_node = graphs.ProgramNode(1,affinities=vflags)

        parents = [generate_tree(layers - 1, branching, compute, data) for i in range(branching)]

        parent_trgs, parent_nodes, parent_edges = zip(*parents)

        dependency_edges = [graphs.ProgramEdge(c_node.id, n_node.id, data_size=data) for n_node in parent_trgs if n_node is not None]

        return c_node, [c_node] + list([item for sublist in parent_nodes for item in sublist]), list(dependency_edges) + list([item for sublist in parent_edges for item in sublist])

    _, p_nodes, p_edges = generate_tree(layers, branching, compute, data, vflags)
    return graphs.ProgramGraph(p_nodes, p_edges)