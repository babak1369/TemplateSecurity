import numpy as np
import garbledcircuit as gc
import time

A = np.array([0,23])
dimension = 2
precision = 10
security = 100
ts = gc.TemplateSecurity(A,precision,4)
wires,et,keys,A,square_sum_query,current,gc_euc =  ts.euclidean_distance_setup()
keys1 = keys[0:dimension*precision]+keys[square_sum_query[1,0]:square_sum_query[1,0]+2*precision]
query = np.array([0,25])


wires,B = ts.prepare_query(query,square_sum_query,wires,keys1)

t= time.time()
wi= gc_euc.degarbling(et,wires)
print(time.time()-t)
np.savetxt('euc_circuit', gc_euc.circuit,fmt="%d")