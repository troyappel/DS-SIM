from abc import ABC, abstractmethod
import functools
import graphs
from utils import *

@functools.total_ordering
class Event(ABC):
    def __init__(self, time, pg: graphs.ProgramGraph, mg: graphs.MachineGraph):
        self.uid = str(gensym())
        self.time = time

        self.pg = pg
        self.mg = mg

    @abstractmethod
    def transform_graphs(self):
        pass

    def __hash__(self):
        return self.uid

    def __int__(self):
        return int(self.uid)
    
    def __lt__(self, other): 
        return self.time < other.time

    def __index__(self):
        return int(self)

# Event for when ALL dependencies of a task are retrieved
class TransferFinishEvent(Event):
    __hash__ = Event.__hash__

    def __init__(self, time, pg: graphs.ProgramGraph, mg: graphs.MachineGraph, task, machine, data):
        super().__init__(time, pg, mg)
        self.machine: graphs.MachineNode = machine
        self.task: graphs.ProgramNode = task
        self.data = data

    def transform_graphs(self):
        self.task.state = graphs.NodeState.READY
        # The machine who requested a fetch now has all the data associated 
        # with this transfer
        self.machine.ready_inputs.update(self.data)

class TaskFinishEvent(Event):
    __hash__ = Event.__hash__

    def __init__(self, time, pg : graphs.ProgramGraph, mg: graphs.MachineGraph, task: graphs.ProgramNode, machine: graphs.MachineNode):
        super().__init__(time, pg, mg)

        self.task = task
        self.machine = machine

    def transform_graphs(self):
        self.machine.stored_outputs.add(self.task)
        self.task.state = graphs.NodeState.COMPLETED
        self.machine.task = None

