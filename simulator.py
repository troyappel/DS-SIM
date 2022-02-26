from queue import PriorityQueue
import events

class Simulator: 
    def __init__(self, mg, pg):
        self.mg = mg 
        self.pg = pg
        self.event_queue = PriorityQueue()

    def _process_event(self, event : events.Event): 
        event.transform_graphs(self.pg, self.mg)
    
    def run(self): 
        pass