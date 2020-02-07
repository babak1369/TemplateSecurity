import numpy as np
def matricize(v,direction=True):

    s =  v.size
    if direction:
        return np.reshape(v,(s,1))
    else:
        return np.reshape(v,(1,s))
class wire:
    def __init__(self,matrix):
        self.matrix= matrix

