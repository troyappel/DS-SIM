from graphs import *
from simulator import *
import generate_programs
import generate_networks
from process_outputs import *

tree_pg = generate_programs.generate_tree_program_graph(layers=3,    # How many levels in the tree 
                                                        branching=2) # How many children of each node

sort_pg = generate_programs.generate_mapreduce(num_map=25,          # Number of reducer tasks
                                               num_reduce=10,       # Number of mapper tasks
                                               map_compute=1,       # Amount of compute required for a single map
                                               reduce_compute=1,    # Above, but for reducer
                                               map_input_sz=1,      # Size of data, in GB for the input data to a map task
                                               map_output_sz=1,     # Size of data out of each map task, and to reducer
                                               reduce_output_sz=1)  # Output from each reducer

grep_pg = generate_programs.generate_mapreduce(num_map=25,          # Number of reducer tasks
                                               num_reduce=1,        # Number of mapper tasks
                                               map_compute=1,       # Amount of compute required for a single map
                                               reduce_compute=1,    # Above, but for reducer
                                               map_input_sz=1,      # Size of data, in GB for the input data to a map task
                                               map_output_sz=0.01,  # Size of data out of each map task, and to reducer
                                               reduce_output_sz=0.01)  # Output from each reducer

dask_pg = generate_programs.generate_dask(layers=2,          # "Layers" of sets of task
                                          num_per_layer=4,   # Tasks in each layer
                                          density=0.5,       # What percentage of dependencies (connect)
                                          min_compute=1,     # Compute per task is randomly selected between this value 
                                          max_compute=4,     # and this one
                                          data=1)            # Output size of every task


datacenter_mg = generate_networks.generate_tree_machine_network(total_machines=8,    # Machines in the cluster
                                                                machines_per_rack=2, # Racks in the cluster, with total_machines/machines_per_rack machines each
                                                                compute_per_core=1,  # Compute/second handled by the machine
                                                                cores=1,             # Cores in each machine
                                                                disk_bandwidth=1,  # Bandwidth between each machine and its disk, if use_disk is true
                                                                rack_bandwidth=1,    # Bandwidth on the machine <--> switch links 
                                                                root_bandwidth=10,   # Bandwidth on the rack switch <--> root switch links
                                                                rack_agg=False,      # If true, then bandwidth is total for the switch. Else per NIC
                                                                root_agg=False,      # If true, then bandwidth is total for the root. Else per rack
                                                                use_disk=True)       # Create disk fileserver machines, with no compute ability. Each machine gets one
                                                                                     # This models local storage

# Dump the output
outfilename = "tree_test.pickle"

# Create a simulation object over the two graphs
tree_task_simulator = Simulator(datacenter_mg, sort_pg)

tree_results = tree_task_simulator.run(speedup = 5,              # If draw_visualization is True, display at speedup x realtime 
                                       outfile = outfilename,    # If not None, full history of the simulation is written here
                                       draw_visualization=False, # Play the visualization after the simulation finishes
                                       print_time=False)         # Print the time to the console at each event. Useful for debugging

# If you wrote to a file, you can pull the results back into memory
mapreduce_sort_data_from_disk = load_history(outfilename)

# -------------------------
# -------Analysis----------
# -------------------------

# We can replay a simulation 
visualize_history(mapreduce_sort_data_from_disk,   # History to replay
                  speedup=5,                # Display at speedup x realtime 
                  merge_frames_window=0.01) # Combine frames whose time falls within merge_frames_window of each other to one frame


# We can create a graph of the history 
mr_sort_history_dataframe = history_df(mapreduce_sort_data_from_disk)
print(mr_sort_history_dataframe.columns)

# If this is mapreduce data, we can make a nice plot from it
plot_mapreduce_data_transfer(mr_sort_history_dataframe)
