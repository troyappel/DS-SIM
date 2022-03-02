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

    def __init__(self, start_time, end_time, pg: graphs.ProgramGraph, mg: graphs.MachineGraph, task, machine, data):
        super().__init__(start_time, end_time, pg, mg)
        self.machine: graphs.MachineNode = machine
        self.task: graphs.ProgramNode = task
        self.data = data

    def transform_graphs(self):
        self.task.state = graphs.NodeState.READY
        # The machine who requested a fetch now has all the data associated 
        # with this transfer
        self.machine.ready_inputs.update(self.data)

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

