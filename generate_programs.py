import graphs

def generate_tree_program_graph(layers=4, branching=3, compute=4, data=1):
    if layers == 0:
        return [], []

    # add coalescing node for this layer
    c_node = graphs.ProgramNode(1)

    parents = [generate_tree_program_graph(layers - 1, branching, compute, data) for i in range(branching)]

    parent_nodes, parent_edges = zip(*parents)

    dependency_edges = [ProgramEdge(0,2, data_size=1) for ]

    p_children

