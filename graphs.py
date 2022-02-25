import networkx as nx
import matplotlib.pyplot as plt

from enum import Enum

# Get unique symbol.
def gensym():
    gensym.acc += 1
    return gensym.acc
gensym.acc = 0

class NodeState(Enum):
    UNSCHEDULED = 0
    WAITING = 1
    RUNNING = 2
    COMPLETED = 3

def to_id(key):
    if isinstance(key, ProgramNode):
        return key.id
    elif isinstance(key, str):
        return key
    elif isinstance(key, tuple):
        return tuple(to_id(k) for k in key)
    else:
        return str(key)

class ProgramNode(object):
    """
    Representation of Program Node.

    Get this via a ProgramGraph method, or by iterating through it.
    """
    def __init__(self, compute, memory, id=None, affinity=None):
        if id is None:
            self.id = f"pn-{gensym()}-{compute}-{memory}"
        else:
            self.id = str(id)

        self.compute = compute
        self.memory = memory

        self.state = NodeState.UNSCHEDULED
        self.machine = None

        if affinity is None:
            self.affinity = lambda x : True
        else:
            self.affinity = affinity

class ProgramEdge(object):
    """
    Program edge representation.

    Stores IDs only of associated nodes, letting us define edges before the graph.
    Get this by a ProgramGraph method.
    """
    def __init__(self, in_node, out_node, cost=0):
        self.id = (to_id(in_node), to_id(out_node))
        self.in_node = to_id(in_node)
        self.out_node = to_id(out_node)
        self.cost = cost

############ MACHINES

class MachineNode:
    """
    Representation of physical nodes.

    Get this by iterating through/finding neighbors in a MachineGraph.
    """
    def __init__(self, compute, ram, id=None):
        if id is None:
            self.id = f"mn-{gensym()}-{compute}-{ram}"
        else:
            self.id = str(id)

        self.compute = compute
        self.ram = ram

        self.task = None

        self.stored_outputs = set()

class MachineEdge:
    """
    Representation of edge between physical nodes. Stores ids of the machines, NOT machine nodes.

    Get this by requesting an edge from a MachineGraph.
    """
    def __init__(self, m1, m2, latency=0, bandwidth=10):
        self.id = to_id(m1), to_id(m2)
        self.machine1 = to_id(m1)
        self.machine2 = to_id(m2)
        self.latency = latency
        self.bandwidth = bandwidth


##################
### GRAPHS ###
##################

class ProgramGraph(object):
    """
    Representation of a program graph.

    Thin wrapper around networkx.DiGraph. Lets you add/remove nodes, get neighbors, iterate, etc.
    All methods handle either a ProgramNode object or its ID.
    """
    def __init__(self, nodes=None, edges=None, strict=True):

        self.G = nx.DiGraph()

        self.node_dict = {}
        self.edge_dict = {}

        self.strict = strict

        if nodes is not None:
            self.add_nodes(nodes)

        if edges is not None:
            self.add_edges(edges)

    def validate(self):
        # Check that there are no cycles
        try:
            nx.find_cycle(self.G)
            return False
        except nx.NetworkXNoCycle:
            pass

        # Check that there are no duplicate ids
        if len(self.node_dict.keys()) != len(set(self.node_dict.keys())):
            return False

        return True

    def __getitem__(self, key):
        if isinstance(key, list):
            return [self.node_dict[to_id(_key)] for _key in key]
        else:
            return self.node_dict[to_id(key)]

    def pred(self, key):
        preds = self.G.predecessors(to_id(key))
        return self[preds]

    def succ(self, key):
        succs = self.G.successors(to_id(key))
        return self[succs]

    def in_edges(self, key):
        eds = self.G.in_edges(to_id(key))
        return [self.edge_dict[ed] for ed in eds]

    def out_edges(self, key):
        eds = self.G.out_edges(to_id(key))
        return [self.edge_dict[ed] for ed in eds]

    def add_node(self, pn):
        self.G.add_node(pn.id)
        if self.strict:
            assert(pn.id not in self.node_dict)
        self.node_dict[pn.id] = pn

    def add_nodes(self, pn_list):
        for pn in pn_list:
            self.add_node(pn)

    def add_edge(self, en):

        id1 = to_id(en.in_node)
        id2 = to_id(en.out_node)

        if self.strict:
            assert (id1 != id2)
            assert (id1 in self.node_dict)
            assert (id2 in self.node_dict)
            assert((id1, id2) not in self.node_dict)
            assert((id1, id2) not in self.G.edges)

        self.G.add_edge(id1, id2)
        self.edge_dict[(id1, id2)] = en

    def add_edges(self, en_list):
        for en in en_list:
            self.add_edge(en)

    def draw(self, blocking=True):
        colors = ["#AAAAAA", "#ACC8DC", "#D8315B", "#1E1B18"]
        pos = nx.spring_layout(self.G)

        for ns in NodeState:
            nodes_ns = [n for n in self.G if self.node_dict[n].state == ns]
            nx.draw_networkx_nodes(self.G, pos, nodelist=nodes_ns, node_size=500, node_color=colors[ns.value])

        nx.draw_networkx_labels(self.G, pos)

        # Draw based on cost of edge
        costs = [self.edge_dict[tup].cost for tup in self.G.edges]

        nx.draw_networkx_edges(self.G, pos, self.G.edges, arrows=True, edge_color=costs, edge_cmap=plt.get_cmap('copper'))

        if blocking:
            plt.show()
        else:
            plt.pause(0.001)

    def __iter__(self):
        yield from self.node_dict.values()



class MachineGraph:
    """
    Representation of physical graph.

    Internally represented as an undirected networkx graph, in case we want graph algos.
    Generally works
    """
    def __init__(self, nodes=None, edges=None, strict=True):
        self.G = nx.Graph()

        self.node_dict = {}
        self.edge_dict = {}

        self.strict = strict

        if nodes is not None:
            self.add_nodes(nodes)

        if edges is not None:
            self.add_edges(edges)

    def validate(self):
        # Check that there are no duplicate ids
        if len(self.node_dict.keys()) != len(set(self.node_dict.keys())):
            return False

        return True

    def neighbors(self, key):
        neigh = self.G.neighbors(to_id(key))
        return self[neigh]

    def edges(self, key):
        eds = self.G.edges(to_id(key))
        return [self.edge_dict[ed] for ed in eds]

    def add_node(self, pn):
        self.G.add_node(pn.id)
        if self.strict:
            assert(pn.id not in self.node_dict)
        self.node_dict[pn.id] = pn

    def add_nodes(self, pn_list):
        for pn in pn_list:
            self.add_node(pn)

    def add_edge(self, en):

        id1 = to_id(en.machine1)
        id2 = to_id(en.machine2)

        if self.strict:
            assert (id1 != id2)
            assert (id1 in self.node_dict)
            assert (id2 in self.node_dict)
            assert((id1, id2) not in self.node_dict)
            assert((id1, id2) not in self.G.edges)

        self.G.add_edge(id1, id2)
        self.edge_dict[(id1, id2)] = en

    def add_edges(self, en_list):
        for en in en_list:
            self.add_edge(en)

    def draw(self, blocking=True, linewidth_exp=0.5):
        """
        Draw this graph using matplotlib.

        :param blocking: Whether to block the program while displaying the graph.
        :param linewidth_exp: Exponent to determine line width based on bandwidth, between 0 and 1.
        """
        colors = ["#AAAAAA", "#ACC8DC", "#D8315B", "#1E1B18"]
        taskless_color = "#7788AA"
        pos = nx.spring_layout(self.G)

        taskless = [n for n in self.G if self.node_dict[n].task is None]
        tasked = [n for n in self.G if self.node_dict[n].task is not None]

        for ns in NodeState:
            nodes_ns = [n.task.state for n in tasked if self.node_dict[n].task.state == ns]
            nx.draw_networkx_nodes(self.G, pos, nodelist=nodes_ns, node_size=500, node_color=colors[ns.value])

        nx.draw_networkx_nodes(self.G, pos, nodelist=taskless, node_size=500, node_color=taskless_color)

        nx.draw_networkx_labels(self.G, pos)

        # Draw based on cost of edge
        latencies = [self.edge_dict[tup].latency for tup in self.G.edges]
        bandwidths = [1 + self.edge_dict[tup].bandwidth**linewidth_exp for tup in self.G.edges]

        nx.draw_networkx_edges(self.G, pos, self.G.edges, width=bandwidths,  edge_color=latencies, edge_cmap=plt.get_cmap('copper'))

        if blocking:
            plt.show()
        else:
            plt.pause(0.001)

    def __getitem__(self, key):
        if isinstance(key, list):
            return [self.node_dict[to_id(_key)] for _key in key]
        if isinstance(key, tuple):
            return self.edge_dict[to_id(key)]
        else:
            return self.node_dict[to_id(key)]

    def __iter__(self):
        yield from self.node_dict.values()




# Example usage
# nodes = [ProgramNode(1, 0, 1), ProgramNode(2, 0, 2), ProgramNode(3,0,3)]
# edges = [ProgramEdge(1,2), ProgramEdge(1,3, cost=1), ProgramEdge(2,1)]
# g = ProgramGraph(nodes, edges)
# g.draw()