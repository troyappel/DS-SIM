from abc import  ABC, abstractmethod

import graphs
from utils import *

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

    def __cmp__(self, other):
        return int(self) - int(other)

    def __index__(self):
        return int(self)

# Event for when ALL dependencies of a task are retrieved
class TransferFinishEvent(Event):
    __hash__ = Event.__hash__

    def __init__(self, time, task):
        super().__init__(time)

        self.task_id = graphs.to_id(task) # Task who requested a fetch

    def transform_graphs(self, pg: graphs.ProgramGraph, mg: graphs.MachineGraph):
        pg[self.task_id].state = graphs.NodeState.READY

class TaskFinishEvent(Event):
    __hash__ = Event.__hash__

    def __init__(self, time, task, machine):
        super().__init__(time)

        self.task_id = graphs.to_id(task)
        self.machine_id = graphs.to_id(machine)

    def transform_graphs(self, pg : graphs.ProgramGraph, mg: graphs.MachineGraph):
        pg[self.machine_id].stored_output.add(self.task_id)
        mg[self.task_id].state = graphs.NodeState.COMPLETED

