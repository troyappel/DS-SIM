from abc import  ABC, abstractmethod
import functools
import graphs
from utils import *

@functools.total_ordering
class Event(ABC):
    def __init__(self, time):
        self.uid = str(gensym())
        self.time = time

    @abstractmethod
    def transform_graphs(self, pg: graphs.ProgramGraph, mg: graphs.MachineGraph):
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

    def __init__(self, time, task, machine, data):
        super().__init__(time)
        self.machine_id = graphs.to_id(machine)
        self.task_id = graphs.to_id(task) # Task who requested a fetch
        self.data = data

    def transform_graphs(self, pg: graphs.ProgramGraph, mg: graphs.MachineGraph):
        pg[self.task_id].state = graphs.NodeState.READY
        # The machine who requested a fetch now has all the data associated 
        # with this transfer
        mg[self.machine_id].ready_inputs.update(self.data)

class TaskFinishEvent(Event):
    __hash__ = Event.__hash__

    def __init__(self, time, task, machine):
        super().__init__(time)

        self.task_id = graphs.to_id(task)
        self.machine_id = graphs.to_id(machine)

    def transform_graphs(self, pg : graphs.ProgramGraph, mg: graphs.MachineGraph):
        mg[self.machine_id].stored_output.add(self.task_id)
        pg[self.task_id].state = graphs.NodeState.COMPLETED
        mg[self.machine_id].task = None

