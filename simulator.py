from queue import PriorityQueue
import heapq
from collections import namedtuple
import events
import graphs
from typing import List

# Simple record type to represent a freeze frame of our 
# simulation, for reconstructing data and visualizations
Snapshot = namedtuple('Snapshot', ['time', 'mg', 'pg'])

def time_to_run_task(t, m, eps=1e-5):
    """
    How long does the compute phase of t take to run on m. 
    TODO: Move this somewhere else
    """
    return t.compute / (m.compute + eps)


class Simulator:
    def __init__(self, mg : graphs.MachineGraph, pg : graphs.ProgramGraph):
        self.mg = mg 
        self.pg = pg
        self.event_queue = []
        self.current_time = 0
        self.history : List[Snapshot] = []

    def _start_transfer(self, task, machine):
        """
        Enqueues datatransfer events
        """
        end_time = self.current_time + graphs.get_max_fetch_time(task, machine, self.pg, self.mg)


        # Create a transfer
        for pred_task in self.pg.pred(task):
            time = self.mg.network_distance_real(machine, pred_task.bound_machine, self.pg[(pred_task, task)].data_size)
            ev = events.TransferEvent(self.current_time, self.current_time + time, self.pg, self.mg, task, machine, pred_task)
            heapq.heappush(self.event_queue, ev)

        # The transfer has started, so the associated task is fetching
        self.pg[task].state = graphs.NodeState.FETCHING

        # We will, at this point, definitely run task on machine
        self.pg[task].bound_machine = machine


    def _start_compute(self, task, machine): 

        end_time = self.current_time + time_to_run_task(self.pg[task], self.mg[machine])
        ev = events.TaskEvent(self.current_time, end_time, self.pg, self.mg, task, machine)
        heapq.heappush(self.event_queue, ev)

        # Compute has started; task is running. 
        task.state = graphs.NodeState.RUNNING

        # TODO: Set this here or when the transfer gets started?
        # This will depend on whether we want to support simultaneous 
        # fetch and compute
        machine.task = task

    def _process_event(self, event : events.Event):
        self.current_time = max(self.current_time, event.end_time)
        event.transform_graphs()

        for ev in self.event_queue:
            ev.recalculate_end(self.current_time)

        heapq.heapify(self.event_queue)
        
        if(isinstance(event, events.TransferEvent)):
            # This needs to be made more complicated if we allow 
            # simultaneous fetching and compute (the machine might be busy with)
            # something else when we get here
            self._start_compute(event.task, event.machine)

    def _schedule(self):
        next_up_tasks = self.pg.up_next()
        free_machines = self.mg.idle_machines()

        get_affinity = \
            lambda task, machine: graphs.get_task_affinity(task, machine, self.pg, self.mg)

        # maps task to a descending-order list of (affinity, machine) pairs
        affinities = {
            task : sorted([(get_affinity(task, machine), machine) for machine in free_machines], reverse=True, key=lambda x: x[0])
            for task in next_up_tasks
        }

        # Start as many tasks as you can
        for task, machines in affinities.items():
            # Pick the highest-affinity machine for the task that is still free
            machine_choice = None
            for aff, machine in machines: 
                # Aff = 0 means cannot run
                if machine in free_machines and aff > 0:
                    machine_choice = machine
            # Impossible to run this task
            if machine_choice is None: 
                continue
            
            free_machines.remove(machine_choice)

            if len(self.pg.pred(task)) == 0:
                self.pg[task].bound_machine = machine_choice
                self._start_compute(task, machine_choice)
            else:
                self._start_transfer(task, machine_choice)

    def run(self):

        self.pg.draw(0.01)
        self.mg.draw(5)

        while not self.pg.finished(): 
            # Alternate between processing an event and invoking the scheduler
            # to react to any chances made by that event.
            if len(self.event_queue) > 0:
                self._process_event(heapq.heappop(self.event_queue))
            self._schedule()

            self.history.append(Snapshot(self.current_time, self.mg.snapshot(), self.pg.snapshot()))

            self.pg.draw(0.01)

            try:
                self.mg.draw(self.event_queue[0].end_time - self.current_time + 0.01)
            except IndexError:
                self.mg.draw()
        
        return self.history
