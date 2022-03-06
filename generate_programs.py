import math
import graphs
import random

def generate_type_affinity(type):
    '''
        Returns an affinity function ensuring that
        a task will only be scheduled on nodes with label
        'type'
    '''
    def affinities(x):
        if x == type:
            return 1
        else:
            return 0
    return affinities

def generate_tree_program_graph(layers=2, branching=3, compute=4, data=1, affinities=None):

    def generate_tree(layers=2, branching=3, compute=4, data=1, affinities=None):
        if layers == 0:
            return None, [], []

        if affinities is None:
            affinities = generate_type_affinity('machine')

        # add coalescing node for this layer
        c_node = graphs.ProgramNode(1,affinities=affinities)

        parents = [generate_tree(layers - 1, branching, compute, data) for i in range(branching)]

        parent_trgs, parent_nodes, parent_edges = zip(*parents)

        dependency_edges = [graphs.ProgramEdge(c_node.id, n_node.id, data_size=data) for n_node in parent_trgs if n_node is not None]

        return c_node, [c_node] + list([item for sublist in parent_nodes for item in sublist]), list(dependency_edges) + list([item for sublist in parent_edges for item in sublist])

    _, p_nodes, p_edges = generate_tree(layers, branching, compute, data, affinities)
    return graphs.ProgramGraph(p_nodes, p_edges)

def generate_mapreduce(num_map, num_reduce, map_compute, reduce_compute, map_input_sz, map_output_sz, reduce_output_sz):

    '''
        Generates a mapreduce computation graph for an arbitrary number of mappers/reducers, compute costs, and 
        amount of data transferred
    '''

    dist_func = lambda d: 1/(1+d**2)

    # mapper nodes
    mapper_disk_input = [graphs.ProgramNode(0, id=f'map_input_{i}', 
                         affinities=generate_type_affinity('disk'), dist_affinity=dist_func) for i in range(num_map)]
    mappers = [graphs.ProgramNode(map_compute, id=f'map_{i}', 
               affinities=generate_type_affinity('machine'), dist_affinity=dist_func) for i in range(num_map)]
    mapper_disk_output = [graphs.ProgramNode(0, id=f'map_output_{i}', 
                        affinities=generate_type_affinity('disk'), dist_affinity=dist_func) for i in range(num_map)]

    # reducer nodes
    reducers = [graphs.ProgramNode(reduce_compute, id=f'reduce_{i}', 
               affinities=generate_type_affinity('machine'), dist_affinity=dist_func) for i in range(num_reduce)]
    reducer_disk_output = [graphs.ProgramNode(0, id=f'reduce_output_{i}', 
                          affinities=generate_type_affinity('disk'), dist_affinity=dist_func) for i in range(num_reduce)]

    # mapper input/output edges
    mapper_in = [graphs.ProgramEdge(f'map_input_{i}', f'map_{i}', data_size=map_input_sz) for i in range(num_map)]
    mapper_out = [graphs.ProgramEdge(f'map_{i}', f'map_output_{i}', data_size=map_output_sz) for i in range(num_map)]

    # mapper to reducer edges
    mapper_to_reducer = [graphs.ProgramEdge(f'map_output_{i}', f'reduce_{j}', map_output_sz) for i in range(num_map) for j in range (num_reduce)]

    # reducer output edges
    reducer_out = [graphs.ProgramEdge(f'reduce_{i}', f'reduce_output_{i}', data_size=reduce_output_sz) for i in range(num_reduce)]

    p_nodes = mapper_disk_input + mappers + mapper_disk_output + reducers + reducer_disk_output
    p_edges = mapper_in + mapper_out + mapper_to_reducer + reducer_out
    return graphs.ProgramGraph(p_nodes, p_edges)

def generate_dask(layers=2, num_per_layer=4, density=0.5, min_compute=1, max_compute=4, data=1, affinities=None):
    assert(density <= 1)
    
    dist_func = lambda d: 1/(1+d**2)

    nodes = []
    edges = []
    for layer_num in range(layers):
        layer_compute = random.randint(min_compute, max_compute)

        for node_num in range(num_per_layer):
            node_id = f'{layer_num}-{node_num}'
            nodes.append(graphs.ProgramNode(layer_compute, id=node_id, dist_affinity=dist_func))

            if layer_num < layers - 1:
                for edge_id in random.sample(range(num_per_layer), math.ceil(density * num_per_layer)):
                    edges.append(graphs.ProgramEdge(node_id, f'{layer_num+1}-{edge_id}', data_size=data))
    
    return graphs.ProgramGraph(nodes, edges)


# pg = generate_mapreduce(2, 1, 1, 1, 1, 1, 1)
# pg.draw()

pg = generate_dask(4,3,0.4,1,6)
pg.draw()