import heapq
import events
import graphs
from typing import List
import pickle
from process_outputs import prepare_snapshot_list, visualize_history
from utils import Snapshot
import time

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

        assert task.state == graphs.NodeState.UNSCHEDULED
        assert task.bound_machine is None

        # Create a transfer
        for pred_task in self.pg.pred(task):
            path, time = self.mg.network_distance_real(machine, pred_task.bound_machine, self.pg[(pred_task, task)].data_size)
            assert time >= 0
            ev = events.TransferEvent(self.current_time, self.current_time + time, self.pg, self.mg, task, machine, pred_task)
            heapq.heappush(self.event_queue, ev)

        # The transfer has started, so the associated task is fetching
        task.state = graphs.NodeState.FETCHING

        # We will, at this point, definitely run task on machine
        self.pg[task].bound_machine = machine
        # machine.task = task

    def _start_compute(self, task, machine):

        # print(f'Task: {task.id} scheduled on {machine.id}')
        assert task.state != graphs.NodeState.RUNNING
        assert task.state != graphs.NodeState.COMPLETED

        assert task.state != graphs.NodeState.FETCHING

        assert task.state == graphs.NodeState.UNSCHEDULED or task.state == graphs.NodeState.READY

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

    def _start_ready(self):
        ready_tasks = [p for p in self.pg if p.state == graphs.NodeState.READY]

        for task in ready_tasks:
            if task.bound_machine.is_free():
                self._start_compute(task, task.bound_machine)


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
                    break

            # Impossible to run this task
            if machine_choice is None: 
                continue
            
            free_machines.remove(machine_choice)

            if len(self.pg.pred(task)) == 0:
                self.pg[task].bound_machine = machine_choice
                self._start_compute(task, machine_choice)
            else:
                assert task.state == graphs.NodeState.UNSCHEDULED
                self._start_transfer(task, machine_choice)

    def _write_history(self, filename): 
        try: 
            result = pickle.dumps(self.history)
            # print(result)
            with open(filename, "wb") as output:
                output.write(result)
            return True
        except: 
            return False

    def run(self, speedup = 5, outfile = None, draw_visualization=True, print_time=False):
        start_time = time.time()
        self.history.append(Snapshot(self.current_time, self.mg.snapshot(), self.pg.snapshot(), pickle.dumps([])))
        while not self.pg.finished(): 

            # Alternate between processing an event and invoking the scheduler
            # to react to any chances made by that event.

            serialized_q = pickle.dumps([e for e in self.event_queue])

            next_event = None
            if len(self.event_queue) > 0:
                next_event = heapq.heappop(self.event_queue)
                self._process_event(next_event)
            
            self._start_ready()
            self._schedule()

            self.history.append(Snapshot(self.current_time, self.mg.snapshot(), self.pg.snapshot(), serialized_q))
            if(print_time): 
                print(self.current_time)

        if outfile is not None: 
            if self._write_history(outfile): 
                print(f"Wrote file to {outfile}")
            else: 
                print("Writing history failed")
        
        if draw_visualization:
            visualize_history(prepare_snapshot_list(self.history), speedup=speedup)

        print("--- Simulation took %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        result = prepare_snapshot_list(self.history)
        print("--- Preparing snapshots took %s seconds ---" % (time.time() - start_time))
        return result 
