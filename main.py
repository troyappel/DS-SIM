from graphs import *
from simulator import *
import generate_programs
import generate_networks
from process_outputs import *

# Generate simple inputs
tree_pg = generate_programs.generate_tree_program_graph(layers=3,branching=5)
datacenter_mg = generate_networks.generate_tree_machine_network(8,2,1,1,1,1,2, use_disk=False)

# Dump the output file here
outfilename = "generaltest.pickle"

# Run without drawing the visualization, and write to a file
Simulator(datacenter_mg, tree_pg).run(draw_visualization=True, outfile=outfilename)

# Pull the history object out of the file
reloaded_history = load_history(outfilename)

# Now can visualize again from the file
# visualize_history(reloaded_history)