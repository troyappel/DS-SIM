from graphs import *
from simulator import *
import generate_programs
import generate_networks
from process_outputs import *


# standard (1) and limited (0.1) bandwidth
for rack_bandwidth in [0.1, 1]:
    sort_pg = generate_programs.generate_mapreduce(num_map=25,          
                                               num_reduce=10,      
                                               map_compute=1,      
                                               reduce_compute=1,    
                                               map_input_sz=1,     
                                               map_output_sz=1,    
                                               reduce_output_sz=1)  

    grep_pg = generate_programs.generate_mapreduce(num_map=25,        
                                                num_reduce=1,     
                                                map_compute=1,     
                                                reduce_compute=1,  
                                                map_input_sz=1,      
                                                map_output_sz=0.01,  
                                                reduce_output_sz=0.01)  

    datacenter_mg = generate_networks.generate_tree_machine_network(total_machines=4, machines_per_rack=2, compute_per_core=1,  cores=1,            
                                                                    disk_bandwidth=1,rack_bandwidth=rack_bandwidth, root_bandwidth=10,  
                                                                    rack_agg=False, root_agg=False, use_disk=True) 
    datacenter_mg_2 = generate_networks.generate_tree_machine_network(total_machines=4, machines_per_rack=2, compute_per_core=1,  cores=1,            
                                                                disk_bandwidth=1,rack_bandwidth=rack_bandwidth, root_bandwidth=10,  
                                                                rack_agg=False, root_agg=False, use_disk=True)        
                                                                                        
    sort_simulator = Simulator(datacenter_mg, sort_pg)
    grep_simulator = Simulator(datacenter_mg_2, grep_pg)

    sort_results = sort_simulator.run(speedup = 5, outfile = f'sort_bandwidth{rack_bandwidth}.pickle', draw_visualization=False, print_time=False)   
    grep_results = grep_simulator.run(speedup = 5, outfile = f'grep_bandwidth{rack_bandwidth}.pickle', draw_visualization=False, print_time=False)  
    
    plot_mapreduce_data_transfer(history_df(load_history(f'sort_bandwidth{rack_bandwidth}.pickle')), 
                                 name=f'transfer_sort_bandwidth{rack_bandwidth}.png')
    plot_mapreduce_data_transfer(history_df(load_history(f'grep_bandwidth{rack_bandwidth}.pickle')), 
                                name = f'transfer_grep_bandwidth{rack_bandwidth}.png')
