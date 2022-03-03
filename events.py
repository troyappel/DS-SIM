from abc import ABC, abstractmethod
import functools
import graphs
from utils import *

@functools.total_ordering
class Event(ABC):
    def __init__(self, start_time, end_time, pg: graphs.ProgramGraph, mg: graphs.MachineGraph):
        self.uid = str(gensym())
        self.start_time = start_time
        self.end_time = end_time

        self.pg = pg
        self.mg = mg

    @abstractmethod
    def transform_graphs(self):
        pass

    # Todo: add recomputation of end time triggered by graph alterations. Use to model congestion.
    def recalculate_end(self, time):
        pass

    def __hash__(self):
        return self.uid

    def __int__(self):
        return int(self.uid)
    
    def __lt__(self, other): 
        return self.end_time < other.end_time

    def __index__(self):
        return int(self)


# Event for when ALL dependencies of a task are retrieved
class TransferEvent(Event):
    __hash__ = Event.__hash__

    created_transfers = set()

    def __init__(self, start_time, end_time, pg: graphs.ProgramGraph, mg: graphs.MachineGraph, task, machine, data):
        super().__init__(start_time, end_time, pg, mg)
        self.machine: graphs.MachineNode = machine
        self.task: graphs.ProgramNode = task
        self.prev_task = data

        self.p_edge = self.pg[(self.prev_task, self.task)]

        self.prev_machine = self.prev_task.bound_machine

        self.transfer_progress = 0
        self.used_bandwidth = 0

        self.last_time = self.start_time

        self.path, _ = self.mg.network_distance_real(self.prev_machine, self.machine, self.p_edge.data_size)

        self.recalculate_end(self.start_time)

        assert (self.p_edge) not in self.created_transfers
        assert self.task.state != graphs.NodeState.RUNNING
        self.created_transfers.add(self.p_edge)

    def transform_graphs(self):
        # The machine who requested a fetch now has all the data associated
        # with this transfer
        self.machine.ready_inputs.update([self.prev_task])

        # Released used bandwidth
        self.mg.alter_path_bandwidth(self.path, self.used_bandwidth)

        all_preds = set([p for p in self.pg.pred(self.task)])

        if all_preds.issubset(self.machine.ready_inputs):
            assert self.task.state != graphs.NodeState.COMPLETED
            assert self.task.state != graphs.NodeState.RUNNING
            self.task.state = graphs.NodeState.READY

    def recalculate_end(self, time):
        # Calculate amount transferred based on bandwidth
        dt = time - self.last_time

        eps = 1e-4

        if self.machine == self.prev_machine:
            return

        assert self.task.state != graphs.NodeState.COMPLETED
        assert self.task.state != graphs.NodeState.RUNNING

        self.transfer_progress += dt * self.used_bandwidth
        assert self.transfer_progress < self.p_edge.data_size # Should not finish just by recalculation

        # Undo change in bandwidth
        self.mg.alter_path_bandwidth(self.path, self.used_bandwidth)

        self.used_bandwidth = self.mg.get_path_bandwidth(self.path)

        self.mg.alter_path_bandwidth(self.path, -self.used_bandwidth)

        self.end_time = self.last_time + (self.p_edge.data_size - self.transfer_progress) / (self.used_bandwidth + eps)

        self.last_time = time

class TaskEvent(Event):
    __hash__ = Event.__hash__

    def __init__(self, start_time, end_time, pg : graphs.ProgramGraph, mg: graphs.MachineGraph, task: graphs.ProgramNode, machine: graphs.MachineNode):
        super().__init__(start_time, end_time, pg, mg)

        self.task = task
        self.machine = machine

    def transform_graphs(self):
        self.machine.stored_outputs.add(self.task)
        self.task.state = graphs.NodeState.COMPLETED
        self.machine.task = None

