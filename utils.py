from collections import namedtuple

# Get unique symbol.
def gensym():
    gensym.acc += 1
    return gensym.acc
gensym.acc = 0

def approx_equal(a, b, epsilon): 
    return abs(a - b) < epsilon

# Simple record type to represent a freeze frame of our 
# simulation, for reconstructing data and visualizations
Snapshot = namedtuple('Snapshot', ['time', 'mg', 'pg'])