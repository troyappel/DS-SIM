from functools import reduce
from unittest import skip

import events
from graphs import *
import pickle
from typing import List
from utils import approx_equal, Snapshot
from events import Event

import networkx as nx

import pandas as pd

def prepare_snapshot_list(snapshot_list): 

    def unpickle_snapshot(s : Snapshot): 
        return Snapshot(s.time, pickle.loads(s.mg), pickle.loads(s.pg), pickle.loads(s.eventqueue))

    return [unpickle_snapshot(s) for s in snapshot_list]

def load_history(filename : str):
    print(f"Reading history from {filename}\n")
    with open(filename, "rb") as in_file: 
        data = in_file.read()
    snapshot_list = pickle.loads(data)

    result = prepare_snapshot_list(snapshot_list)
    print(f"Finished loading history from {filename}\n")
    return result
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
        _, mg, pg, _ = history[0]  
        pg_pos = pg.draw(0.0001)
        mg_pos = mg.draw(3)

    current_time = 0
    for snapshot in history: 
        time, mg, pg, _ = snapshot 

        pg_pos = pg.draw(0.0001, pos_override=pg_pos)
        
        try:
            t = (time - current_time) / speedup
            mg.draw(max(t, 0.01), pos_override=mg_pos)
            current_time = time
        except IndexError:
            mg_pos = mg.draw(pos_override=mg_pos)
    
        if print_time:
            print(f"Time: {round(current_time, 2)}")

def sum_dicts(d_list):
    def reducer(accumulator, element):
        for key, value in element.items():
            accumulator[key] = accumulator.get(key, 0) + value
        return accumulator
    return reduce(reducer, d_list, {})

def generation_dict(eventqueue, generation):
    data = {'bw_in': 0., 'bw_out': 0., 'trans_in_ct': 0., 'trans_out_ct': 0., 'run_ct': 0.}
    for e in eventqueue:
        if isinstance(e, events.TransferEvent):
            if e.task.id in generation:
                data['bw_in'] += e.used_bandwidth
                data['trans_in_ct'] += 1
            elif e.prev_task.id in generation:
                data['bw_out'] += e.used_bandwidth
                data['trans_out_ct'] += 1

        if isinstance(e, events.TaskEvent):
            if e.task.id in generation:
                data['run_ct'] += 1

    return data

def append_to_key(d, i):
    for key in list(d):
        d[f'{key}_{i}'] = d[key]
        del d[key]

def get_step_data(eventqueue, time, generations):
    g_dicts = [generation_dict(eventqueue, gen) for gen in generations]
    sum_dict = sum_dicts(g_dicts)

    step_dict = {'time': time}
    step_dict.update(sum_dict)

    for i, d in enumerate(g_dicts):
        append_to_key(d, i)
        step_dict.update(d)

    return step_dict

def history_df(history):

    if len(history) == 0:
        return None

    _, mg0, pg0, _ = history[0]

    generations = list(nx.topological_generations(pg0.G))

    ds = []

    for snapshot in history:
        time, mg, pg, eventqueue = snapshot

        ds.append(get_step_data(eventqueue, time, generations))

    return pd.DataFrame(ds)

def plot_mapreduce_data_transfer(hist_df):
    '''
        Plot mapper input, mapper to reducer (shuffle), and reducer output data transfer over time
    '''
    t = hist_df['time'].to_list()
    input = hist_df['bw_out_0'].to_list()
    shuffle = hist_df['bw_out_2'].to_list()
    output = hist_df['bw_out_3'].to_list()
    plt.plot(t, input, label='Input')
    plt.plot(t, shuffle, label='Map to Reduce')
    plt.plot(t, output, label='Output')
    plt.xlabel("Timestep")
    plt.ylabel("Data/Time")
    plt.legend(loc='upper right')
    plt.show()