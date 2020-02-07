import numpy as np


""""
PARALLEL
"""
A = np.array([0,23,24,24])
A = np.ones((200))
dimension = 200
precision = 10
security = 100
ts = TemplateSecurity(A,precision,4)
wires_mult,wires_euc,et_list,et_euc,gc_euc,list_of_gc, keys_euc,keys_list_mult,square_sum_query,current,m = ts.parallel_euc_setup()
available_keys_euc = keys_euc[square_sum_query[1,0]:square_sum_query[1,0]+2*precision]
mult_keys_np = np.array(keys_list_mult)
available_keys_mult = mult_keys_np[:,precision:2*precision,:]
query = np.array([0,25,23,4])
query = np.zeros((200))
wires_euc,wires_mult = ts.parallel_prepare_query(query,square_sum_query,wires_euc,wires_mult,available_keys_euc,available_keys_mult)
t = time.time()
wi = group_degarbling_(np.array(et_list),wires_mult,list_of_gc[0].circuit,list_of_gc[0].security)

wires_euc[0:2*precision*dimension,:] = wi[:,m.matrix[1,:]].reshape((2*precision*dimension,security+1))
wires = degarbling_(et_euc,wires_euc,gc_euc.circuit,security)
print(time.time()-t)
"""
end of parallel
"""
wires,et,keys,A,square_sum_query,current,gc_euc =  ts.euclidean_distance_setup()
keys1 = keys[0:dimension*precision]+keys[square_sum_query[1,0]:square_sum_query[1,0]+2*precision]
query = np.array([0,25,23])


wires,B = ts.prepare_query(query,square_sum_query,wires,keys1)

t= time.time()
wi= gc_euc.degarbling(et,wires)
print(time.time()-t)
