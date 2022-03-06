from graphs import *
from simulator import *
import generate_programs
import generate_networks
from process_outputs import *

# Generate simple inputs
tree_pg = generate_programs.generate_tree_program_graph(layers=4,branching=5)
datacenter_mg = generate_networks.generate_tree_machine_network(16,2,1,1,1,1,2, use_disk=False)

# Example that caused crash: 
# tree_pg = generate_programs.generate_mapreduce(40, 8, 1, 1, 1, 1, 1)
# datacenter_mg = generate_networks.generate_tree_machine_network(4,2,1,1,1,1,2, use_disk=True)

# Example that caused crash: 
# tree_pg = generate_programs.generate_mapreduce(40, 8, 1, 1, 1, 1, 1)
# datacenter_mg = generate_networks.generate_tree_machine_network(4,2,1,1,1,1,2, use_disk=True)

# Dump the output file here
outfilename = "generaltest.pickle"

# Run without drawing the visualization, and write to a file
in_memory_history = Simulator(datacenter_mg, tree_pg).run(draw_visualization=False, outfile=outfilename)

# Pull the history object out of the file
reloaded_history = load_history(outfilename)

# # Note that reloaded_history and in_memory_history are equivalent

<<<<<<< HEAD
# Now can visualize again from these representations
visualize_history(in_memory_history, speedup=5, merge_frames_window=.01)
=======
# # Now can visualize again from these representations
visualize_history(in_memory_history, print_time=False)
>>>>>>> 085e48a4c788f9e4d627d5b8974dc5585ce045c7
