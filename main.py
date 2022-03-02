from graphs import *
from simulator import *
import generate_programs
import generate_networks

_, p_nodes, p_edges = generate_programs.generate_tree_program_graph(layers=3)
pg = ProgramGraph(p_nodes, p_edges)


mg = generate_networks.generate_tree_machine_network(8,2,1,1,1,1,1, use_disk=False)

s = Simulator(mg, pg)
s.run()