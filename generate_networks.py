from graphs import *

def generate_tree_machine_network(total_machines, machines_per_rack, compute_per_core, cores,  
                                  disk_bandwidth, rack_bandwidth, root_bandwidth, rack_agg=False, root_agg=False, use_disk=True):
    """
    Generates a 2-level two-level tree-shaped switched network
    (the cluster configuration in the MapReduce paper)
    Note: rack_agg and root_agg are each True if the rack/root bandwidth input
    is expressed as an aggregate or from point to point
    """
    if total_machines % machines_per_rack != 0:
        raise ValueError("Nodes must be evenly divided amongst racks")
    
    num_racks = total_machines//machines_per_rack

    # generate nodes
    machines = [MachineNode(compute_per_core * cores, f'machine_{i}', {'machine'}) for i in range(total_machines)]
    rack_switches = [MachineNode(0, f'rack_switch_{i}', {'rack_switch'}) for i in range(num_racks)]
    root_switch = MachineNode(0, 'root_switch', {'root_switch'})
    if use_disk:
        disks = [MachineNode(0, f'disk_{i}', {'disk'}) for i in range(total_machines)]
    else:
        disks = []
    nodes = machines + rack_switches + [root_switch] + disks


    if rack_agg:
        rack_bandwidth = rack_bandwidth//(2 * machines_per_rack)
    if root_agg:
        root_bandwidth = root_bandwidth//(2 * num_racks)
    edges = []
    # connect machines to racks
    for i in range(total_machines):
        edges.append(MachineEdge(f'machine_{i}', f'rack_switch_{i//num_racks}', bandwidth=rack_bandwidth))

    # connect racks to root
    for i in range(num_racks):
        edges.append(MachineEdge(f'rack_switch_{i}', 'root_switch', bandwidth=root_bandwidth))

    # connect machines to disks
    if use_disk:
        for i in range(total_machines):
            edges.append(MachineEdge(f'machine_{i}', f'disk_{i}', bandwidth=disk_bandwidth))

    return MachineGraph(nodes, edges)

# example - rack/root bandwidths are often (1/10 Gbps), which seems reasonable based on the mapreduce paper
# mg = generate_tree_machine_network(9, 3, 1, 4, 100, 1, 10)
# mg.draw()



