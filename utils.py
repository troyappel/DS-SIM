# Get unique symbol.
def gensym():
    gensym.acc += 1
    return gensym.acc
gensym.acc = 0
