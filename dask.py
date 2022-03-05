from graphs import *
from simulator import *
# from generate_networks import generate_tree_machine_network

from graphs import *


# build machine graph - very simple, three machines, all connected, varied bw

#               0
#              / \
#             1 - 2

m_nodes =  [MachineNode(1, id=0, labels={'machine'}),
            MachineNode(1, id=1, labels={'machine'}),
            MachineNode(1, id=2, labels={'machine'})]

m_edges =  [MachineEdge(0, 1),
            MachineEdge(1, 2),
            MachineEdge(2, 0)]

mg = MachineGraph(m_nodes, m_edges)

def dist_func(d):
    return 1/(1+d**2)

# build base program graph from: https://docs.dask.org/en/stable/graphviz.html
p_nodes =  [ProgramNode(1, id=0, dist_affinity=dist_func), # these are the four ones/adds
            ProgramNode(1, id=1, dist_affinity=dist_func), 
            ProgramNode(1, id=2, dist_affinity=dist_func),
            ProgramNode(1, id=3, dist_affinity=dist_func),
            ProgramNode(2, id=4, dist_affinity=dist_func), # these are the two transposes
            ProgramNode(2, id=5, dist_affinity=dist_func)]

p_edges =  [ProgramEdge(0, 4, data_size=2),
            ProgramEdge(0, 3, data_size=2),
            ProgramEdge(2, 1, data_size=2),
            ProgramEdge(2, 5, data_size=2),
            ProgramEdge(5, 3, data_size=2),
            ProgramEdge(4, 1, data_size=2)]

pg = ProgramGraph(p_nodes, p_edges)

# run simulation
s = Simulator(mg, pg)
s.run(outfile="test.txt", draw_visualization=False)
