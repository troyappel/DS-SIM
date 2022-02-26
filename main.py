from graphs import *
from simulator import *

p_nodes = [ProgramNode(1, 0, 1), ProgramNode(2, 1, 2), ProgramNode(3,2,3)]
p_edges = [ProgramEdge(0,1), ProgramEdge(0,2, data_size=1)]
pg = ProgramGraph(p_nodes, p_edges)

m_nodes = [MachineNode(5, 0), MachineNode(3, 1), MachineNode(2, 2)]
m_edges = [MachineEdge(0, 1), MachineEdge(1, 2), MachineEdge(2, 0)]
mg = MachineGraph(m_nodes, m_edges)

s = Simulator(mg, pg)
s.run()