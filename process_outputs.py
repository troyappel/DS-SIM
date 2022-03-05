from unittest import skip
from graphs import *
import pickle
from typing import List
from utils import approx_equal, Snapshot

def prepare_snapshot_list(snapshot_list): 

    def unpickle_snapshot(s : Snapshot): 
        return Snapshot(s.time, pickle.loads(s.mg), pickle.loads(s.pg))

    return [unpickle_snapshot(s) for s in snapshot_list]

def load_history(filename : str):
    with open(filename, "rb") as in_file: 
        data = in_file.read()
    snapshot_list = pickle.loads(data)

    return prepare_snapshot_list(snapshot_list)

def merge_frames(history, merge_frames_window):

    if merge_frames_window is None: 
        return history

    i = 0
    while i < (len(history) - 1):
        if approx_equal(history[i].time, history[i+1].time, merge_frames_window): 
            del history[i]
            i-=1
        i+=1
    return history

def visualize_history(history, speedup=5, print_time=True, merge_frames_window=0): 
    
    history = merge_frames(history, merge_frames_window)
    
    if len(history) != 0:
        _, mg, pg = history[0]  
        pg.draw(0.0001)
        mg.draw(3)

    current_time = 0
    for snapshot in history: 
        time, mg, pg = snapshot 

        pg.draw(0.0001)
        
        try:
            t = (time - current_time) / speedup
            mg.draw(max(t, 0.01))
            current_time = time
        except IndexError:
            mg.draw()
    
        if print_time:
            print(f"Time: {round(current_time, 2)}")
    