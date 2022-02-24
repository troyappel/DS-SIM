import networkx as nx
import matplotlib.pyplot as plt

from enum import Enum

def gensym():
    gensym.acc += 1
    return gensym.acc
gensym.acc = 0

class NodeState(Enum):
    UNSCHEDULED = 0
    WAITING = 1
    RUNNING = 2
    COMPLETED = 3

class ProgramNode(object):
    def __init__(self, compute, memory, id=None, affinity=None):
        if id is None:
            self.id = f"{gensym()}-{compute}-{memory}"
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

class ProgramGraph(object):
    """
    Representation of a program graph.

    Thin wrapper around networkx.DiGraph. Lets you add/remove nodes, get neighbors, iterate,
    and get more info by key. Keys can be a list of nodes, a node, or a node id.
    """
    def __init__(self, nodes=None, edges=None):
        self.node_dict = {pn.id: pn for pn in nodes}
        self.G = nx.DiGraph()

        for pn in list(self.node_dict):
            self.G.add_node(pn)

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
        assert(len(self.node_dict.keys()) == len(set(self.node_dict.keys())))

    @staticmethod
    def _get_id(key):
        if isinstance(key, ProgramNode):
            return key.id
        elif isinstance(key, str):
            return key
        else:
            return str(key)

    def __getitem__(self, key):
        if isinstance(key, list):
            return [self.node_dict[self._get_id(_key)] for _key in key]
        else:
            return self.node_dict[self._get_id(key)]

    def pred(self, key):
        return self.G.predecessors(self._get_id(key))

    def succ(self, key):
        return self.G.successors(self._get_id(key))

    def add_node(self, pn):
        self.G.add_node(pn.id)
        assert(pn.id not in self.node_dict)
        self.node_dict[pn.id] = pn

    def add_edge(self, key1, key2):
        id1 = self._get_id(key1)
        id2 = self._get_id(key2)

        assert (id1 != id2)

        self.G.add_edge(id1, id2)

    def add_edges(self, edge_list):
        for (id1, id2) in edge_list:
            self.add_edge(id1, id2)

    def __iter__(self):
        yield from self.node_dict.values()

    def draw(self, blocking=True):
        colors = ["#AAAAAA", "#ACC8DC", "#D8315B", "#1E1B18"]
        pos = nx.spring_layout(self.G)

        for ns in NodeState:
            nodes_ns = [n for n in self.G if self.node_dict[n].state == ns]
            nx.draw_networkx_nodes(self.G, pos, nodelist=nodes_ns, node_size=500, node_color=colors[ns.value])

        nx.draw_networkx_labels(self.G, pos)

        nx.draw_networkx_edges(self.G, pos, self.G.edges, arrows=False)

        if blocking:
            plt.show()
        else:
            plt.pause(0.001)


