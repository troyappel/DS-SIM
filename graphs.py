from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy

from typing import Callable, List, Tuple, Set, Optional, Dict, Union, Type
import pickle 
import functools
import collections

# added b/c of weird networkx error with python3.10: https://githubmemory.com/index.php/@networkx
import importlib.machinery
import networkx as nx
import matplotlib.pyplot as plt
import functools
import math

from enum import IntEnum

from utils import *

class NodeState(IntEnum):
    _ignore_ = ['table']

    UNSCHEDULED = 0
    FETCHING = 1
    READY = 2
    RUNNING = 3
    COMPLETED = 4

    @property
    def tbl(self):
        return {
            self.UNSCHEDULED: "#F05C4F",
            self.FETCHING: "#E68943",
            self.READY: "#D8315B",
            self.RUNNING: "#FFBF36",
            self.COMPLETED: "#5DC24E",
        }

    def color(self):
        return self.tbl[self]


def to_id(key):
    if isinstance(key, ProgramNode):
        return key.id
    elif isinstance(key, MachineNode):
        return key.id
    elif isinstance(key, str):
        return key
    elif isinstance(key, tuple):
        return tuple(to_id(k) for k in key)
    else:
        return str(key)



class SuperNode(object):
    def __init__(self, id=None):
        if id is None:
            self.id = f"pn-{gensym()}"
        else:
            self.id = str(id)

class SuperEdge(object):
    def __init__(self, in_node, out_node):
        self.id: Tuple[str, str] = (to_id(in_node), to_id(out_node))
        self.in_node: str = to_id(in_node)
        self.out_node: str = to_id(out_node)


class SuperGraph(object):
    def __init__(self, graph, nodes=None, edges=None, strict=True):
        self.G: Union[nx.DiGraph, nx.Graph] = graph

        self.node_dict: Dict[str, Union[SuperNode, ProgramNode, MachineNode]] = {}
        self.edge_dict: Dict[tuple[str, str], Union[SuperEdge, ProgramEdge, MachineEdge]] = {}

        self.strict: bool = strict

        if nodes is not None:
            self.add_nodes(nodes)

        if edges is not None:
            self.add_edges(edges)

        self.pos = None

    def add_node(self, pn):
        self.G.add_node(pn.id)
        if self.strict:
            assert (pn.id not in self.node_dict)
        self.node_dict[pn.id] = pn

        self.pos = None

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
            assert ((id1, id2) not in self.node_dict)
            assert ((id1, id2) not in self.G.edges)

        self.G.add_edge(id1, id2)
        self.edge_dict[(id1, id2)] = en

    def add_edges(self, en_list):
        for en in en_list:
            self.add_edge(en)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.node_dict[key]
        elif isinstance(key, tuple):
            return self.edge_dict[to_id(key)]
        elif isinstance(key, list) or isinstance(key, collections.abc.Iterable):
            return [self.node_dict[to_id(_key)] for _key in key]
        else:
            return self.node_dict[to_id(key)]

    def __iter__(self):
        yield from self.node_dict.values()

    @abstractmethod
    def draw(self, blocking_time=-1, **kwargs):
        pass

    def snapshot(self):
        return pickle.dumps(self)

    @abstractmethod
    def validate(self):
        pass


class ProgramNode(SuperNode):
    """
    Representation of Program Node.

    Get this via a ProgramGraph method, or by iterating through it.
    """
    def __init__(self, compute, id=None, affinities=None, dist_affinity=None):
        super().__init__(id)

        self.compute: int = compute
        self.state: NodeState = NodeState.UNSCHEDULED
        self.bound_machine: Optional[MachineNode] = None

        self.affinities: Callable[[str], float]
        if affinities is None:
            self.affinities = lambda x: 1
        else:
            self.affinities = affinities

        self.dist_affinity: Callable[[int], float]
        if dist_affinity is None:
            self.dist_affinity = lambda x: 1 / (1 + x)
        else:
            self.dist_affinity = dist_affinity

    def __getstate__(self):
        state = self.__dict__.copy()
        # Pickle can't deal with local lambdas, so we 
        # need to skip dist_affinity and affinities
        del state["affinities"]
        del state["dist_affinity"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Add dist_affinity and affinities back since it doesn't exist in the pickle
        self.dist_affinity = None
        self.affinities = None


class ProgramEdge(SuperEdge):
    """
    Program edge representation.

    Stores IDs only of associated nodes, letting us define edges before the graph.
    Get this by a ProgramGraph method.
    """
    def __init__(self, in_node, out_node, data_size=0):
        super().__init__(in_node, out_node)
        self.data_size: int = data_size

class MachineNode(SuperNode):
    """
    Representation of physical nodes.

    Get this by iterating through/finding neighbors in a MachineGraph.
    """
    def __init__(self, compute, id=None, labels=None):
        super().__init__(id)

        self.compute: int = compute
        self.task: Optional[ProgramNode] = None
        self.stored_outputs: Set[ProgramNode] = set()
        self.ready_inputs: Set[ProgramEdge] = set()

        self.labels: Set[str]
        if labels is None:
            self.labels = set()
        else:
            self.labels = labels

    def is_free(self):
        return self.task is None

class MachineEdge(SuperEdge):
    """
    Representation of edge between physical nodes. Stores ids of the machines, NOT machine nodes.

    Get this by requesting an edge from a MachineGraph.
    """
    def __init__(self, in_node, out_node, latency=0, bandwidth=10):
        super().__init__(in_node, out_node)
        self.latency:   float = latency
        self.bandwidth: float = bandwidth


##################
### GRAPHS ###
##################

class ProgramGraph(SuperGraph):
    """
    Representation of a program graph.

    Thin wrapper around networkx.DiGraph. Lets you add/remove nodes, get neighbors, iterate, etc.
    All methods handle either a ProgramNode object or its ID.
    """
    def __init__(self, nodes=None, edges=None, strict=True):
        super().__init__(nx.DiGraph(), nodes, edges, strict)
        self.completed_preds = set()

    ## GRAPH-SPECIFIC FUNCTIONS
    @functools.cache
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
    
    
    def preds_completed(self, node):
        if node in self.completed_preds: 
            return True
        result = all([x.state == NodeState.COMPLETED for x in self.pred(node)])    

        if result: 
            self.completed_preds.add(node)
        
        return result

    def up_next(self): 
        '''
        Returns list of graphs whose predecessors are all complete
        '''


        return [x for x in list(self.node_dict.values()) if self.preds_completed(x) and x.state == NodeState.UNSCHEDULED]
    
    def finished(self): 
        '''
        All nodes processed
        '''
        return all([x.state == NodeState.COMPLETED for x in self])


    def idle_machines(self):
        return set([n for n in self if n.is_free()])


    def draw(self, blocking_time=-1, pos_override=None, **kwargs):
        # colors = ["#AAAAAA", "#0000FF", "#00FFFF", "#00FF00", "#FFFF00"]

        if pos_override is not None: 
            self.pos = pos_override

        if self.pos is None:
            generations = list(nx.topological_generations(self.G))
            for i, g in enumerate(generations):
                for node in g:
                    self.G.nodes[node]['layer'] = i
            self.pos = nx.multipartite_layout(self.G, subset_key='layer')

        plt.figure(0)
        plt.clf()
        all_nodes = []
        all_colors = []
        for ns in NodeState:
            nodes_ns = [n for n in self.G if self.node_dict[n].state == ns]
            colors = [ns.color()]*len(nodes_ns)
            all_nodes += nodes_ns
            all_colors += colors

        nx.draw_networkx_nodes(self.G, self.pos, nodelist=all_nodes, node_size=500, node_color=all_colors)

        nx.draw_networkx_labels(self.G, self.pos)

        # Draw based on cost of edge
        costs = [self.edge_dict[tup].data_size for tup in self.G.edges]

        nx.draw_networkx_edges(self.G, self.pos, self.G.edges, arrows=True, edge_color=costs, edge_cmap=plt.get_cmap('copper'))

        if blocking_time == -1:
            plt.show()
        else:
            plt.pause(blocking_time)

        return self.pos

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




class MachineGraph(SuperGraph):
    """
    Representation of physical graph.

    Internally represented as an undirected networkx graph, in case we want graph algos.
    Generally works
    """
    def __init__(self, nodes=None, edges=None, strict=True):
        super().__init__(nx.DiGraph(), nodes, edges, strict)

    def add_edge(self, en):
        # implicitly create an edge in the other direction. Use same base class implementation
        swapped_en = copy(en)
        swapped_en.in_node, swapped_en.out_node = swapped_en.out_node, swapped_en.in_node
        swapped_en.id = (swapped_en.in_node, swapped_en.out_node)

        super(MachineGraph, self).add_edge(en)
        super(MachineGraph, self).add_edge(swapped_en)


    def neighbors(self, key):
        neigh = self.G.neighbors(to_id(key))
        return self[neigh]

    def edges(self, key):
        eds = self.G.edges(to_id(key))
        return [self.edge_dict[ed] for ed in eds]

    def idle_machines(self): 
        return set([n for n in self if n.is_free()])

    @functools.cache
    def _network_distance_est_id(self, id1, id2):

        # Zero distance to self
        if id1 == id2:
            return 0

        # Annotate graph with weights
        for edge in self.G.edges:
            self.G.edges[edge]['weight'] = 1 / self.edge_dict[edge].bandwidth

        # Get path
        path = nx.shortest_path(self.G, id1, id2, 'weight')
        dist = nx.shortest_path_length(self.G, id1, id2, 'weight')

        assert path[0] == id1
        assert path[-1] == id2

        return path, dist

    def network_distance_est(self, m1, m2):
        return self._network_distance_est_id(to_id(m1), to_id(m2))

    @functools.cache
    def _network_distance_real_id(self, id1, id2, data_size):

        eps = 1e-4

        # Zero distance to self
        if id1 == id2:
            return [], 0

        for edge in self.G.edges:
            w = self.edge_dict[edge].latency + data_size / (self.edge_dict[edge].bandwidth + eps)
            w = max(w, 0)
            self.G.edges[edge]['weight'] = w


        # Get path
        path = nx.shortest_path(self.G, id1, id2, 'weight')
        dist = nx.shortest_path_length(self.G, id1, id2, 'weight')

        assert path[0] == id1
        assert path[-1] == id2


        return path, dist

    def network_distance_real(self, m1, m2, data_size):
        return self._network_distance_real_id(to_id(m1), to_id(m2), data_size)

    # def decrement_along_path

    def get_path_edges(self, path):
        a = [self[(a,b)] for a,b in zip(path[:-1], path[1:])]
        if len(path) > 1:
            assert len(a) == len(path) - 1
        return a

    def get_path_latency(self, path):
        lats = [e.latency for e in self.get_path_edges(path)]

        # sum of all latencies
        return sum(lats)

    def get_path_bandwidth(self, path):
        bws = [e.bandwidth for e in self.get_path_edges(path)]

        # Bottlenecked at smallest bandwidth
        return min(bws, default=0)

    def alter_path_bandwidth(self, path, delta):
        for e in self.get_path_edges(path):
            e.bandwidth += delta
            assert e.bandwidth >= -1e-4


    def draw(self, blocking_time=-1, pos_override=None, **kwargs):
        """
        Draw this graph using matplotlib.

        :param blocking_time: Whether to block the program while displaying the graph.
        :param linewidth_exp: Exponent to determine line width based on bandwidth, between 0 and 1.
        """
        # colors = ["#AAAAAA", "#ACC8DC", "#D8315B", "#1E1B18", "#FF5733"]
        taskless_color = "#7788AA"

        plt.figure(1)
        plt.clf()

        linewidth_exp = kwargs.get("linewidth_exp", 0.5)

        if pos_override is not None: 
            self.pos = pos_override

        if self.pos is None:
            self.pos = nx.spring_layout(self.G)

        taskless = [n for n in self.G if self.node_dict[n].task is None]
        tasked = [n for n in self.G if self.node_dict[n].task is not None]

        for ns in NodeState:
            nodes_ns = [n for n in tasked if self[n].task.state == ns]
            nx.draw_networkx_nodes(self.G, self.pos, nodelist=nodes_ns, node_size=500, node_color=ns.color())

        nx.draw_networkx_nodes(self.G, self.pos, nodelist=taskless, node_size=500, node_color=taskless_color)
        nx.draw_networkx_labels(self.G, self.pos)

        # Draw based on cost of edge
        latencies = [self.edge_dict[tup].latency for tup in self.G.edges]
        bandwidths = [0.5 + self.edge_dict[tup].bandwidth**linewidth_exp for tup in self.G.edges]

        # bandwidths = list(range(len(bandwidths)))
        # print(bandwidths)
        # print(self.G.edges)
        #
        # print(min(bandwidths))

        nx.draw_networkx_edges(self.G, self.pos, self.G.edges, width=bandwidths,  connectionstyle='arc3,rad=0.1')

        if blocking_time == -1:
            plt.show()
        else:
            plt.pause(blocking_time)
        
        return self.pos

    def validate(self):
        # Check that there are no duplicate ids
        if len(self.node_dict.keys()) != len(set(self.node_dict.keys())):
            return False

        return True


def get_max_fetch_time(task, machine, pg: ProgramGraph, mg: MachineGraph, heuristic=False):
    """
    :param heuristic: Whether or not to use actual distance or an inverse-bandwidth heuristic.
    :return: The max time to fetch all previous results, None otherwise
    """
    t_id = to_id(task)
    m_id = to_id(machine)

    preds = pg.pred(t_id)

    preds_machine_ids = [p.bound_machine for p in preds]

    if not pg.preds_completed(t_id): # any(p.state != NodeState.COMPLETED for p in preds):
        return None

    if any(m is None for m in preds_machine_ids):
        return None

    if len(preds) == 0:
        return 0

    # Calculate fetch time for each
    if heuristic:
        times = [mg.network_distance_est(m_id, _m_id)[1] for _m_id in preds_machine_ids]
    else:
        edge_data_sizes = [pg[(_p, t_id)].data_size for _p in preds]
        times = [mg.network_distance_real(m_id, _m_id, sz)[1] for _m_id, sz in zip(preds_machine_ids, edge_data_sizes)]

    # Max across fetch times
    return max(times)

def get_task_affinity(task, machine, pg: ProgramGraph, mg: MachineGraph):
    """
    :return: affinity, calculated by multiplying distance affinity, and affinity for each type of machine this is
    """
    t_id = to_id(task)
    m_id = to_id(machine)

    dist = get_max_fetch_time(task, machine, pg, mg)

    # Get mult of all machine types
    machine_affinities = [pg[t_id].affinities(m_type) for m_type in mg[m_id].labels]

    return pg[t_id].dist_affinity(dist) * math.prod(machine_affinities)



# Example usage
# nodes = [ProgramNode(1, 0, 1), ProgramNode(2, 0, 2), ProgramNode(3,0,3)]
# edges = [ProgramEdge(1,2), ProgramEdge(1,3, cost=1), ProgramEdge(2,1)]
# g = ProgramGraph(nodes, edges)
# g.draw()