from graphs import *

def generate_tree_machine_network(total_machines, machines_per_rack, compute_per_core, cores,  
                                  disk_bandwidth, rack_bandwidth, root_bandwidth, rack_agg=False, root_agg=False):
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
    machines = [MachineNode(compute_per_core * cores, i, {'machine'}) for i in range(total_machines)]
    rack_switches = [MachineNode(0, len(machines) + i, {'rack_switch'}) for i in range(num_racks)]
    root_switch = MachineNode(0, len(machines) + len(rack_switches), {'root_switch'})
    disks = [MachineNode(0, len(machines) + len(rack_switches) + 1 + i, {'disk'}) for i in range(total_machines)]
    nodes = machines + rack_switches + [root_switch] + disks


    if rack_agg:
        rack_bandwidth = rack_bandwidth//(2 * machines_per_rack)
    if root_agg:
        root_bandwidth = root_bandwidth//(2 * num_racks)
    edges = []
    # connect machines to racks
    for i in range(total_machines):
        edges.append(MachineEdge(i, len(machines) + i//num_racks, bandwidth=rack_bandwidth/2))

    # connect racks to root
    for i in range(num_racks):
        edges.append(MachineEdge(len(machines) + i, len(machines) + len(rack_switches), bandwidth=root_bandwidth/2))

    # connect machines to disks
    for i in range(total_machines):
        edges.append(MachineEdge(i, len(machines) + len(rack_switches) + 1 + i, bandwidth=disk_bandwidth))

    return MachineGraph(nodes, edges)

# example
# mg = generate_tree_machine_network(9, 3, 1, 4, 100, 10, 1)
# mg.draw()



