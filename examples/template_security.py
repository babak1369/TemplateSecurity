import numpy as np
import garbledcircuit as gc
import time

"""
in this example we are checking to see if the query query = np.array([0,25,23,4]) is close enough (threshold = 4) 
to the template A = np.array([0,25,24,3])

"""
A = np.array([0,25,24,3])
dimension = 4
precision = 10
security = 100
threshold = 4
ts = gc.TemplateSecurity(A,precision,threshold)
wires_mult,wires_euc,et_list,et_euc,gc_euc,list_of_gc, keys_euc,keys_list_mult,square_sum_query,current,m = ts.parallel_euc_setup()
available_keys_euc = keys_euc[square_sum_query[1,0]:square_sum_query[1,0]+2*precision]
mult_keys_np = np.array(keys_list_mult)
available_keys_mult = mult_keys_np[:,precision:2*precision,:]
query = np.array([0,25,23,4])
wires_euc,wires_mult = ts.parallel_prepare_query(query,square_sum_query,wires_euc,wires_mult,available_keys_euc,available_keys_mult)
t = time.time()
wi = gc.group_degarbling_(np.array(et_list),wires_mult,list_of_gc[0].circuit,list_of_gc[0].security)

wires_euc[0:2*precision*dimension,:] = wi[:,m.matrix[1,:]].reshape((2*precision*dimension,security+1))
wires = gc.degarbling_(et_euc,wires_euc,gc_euc.circuit,security)
print("computation time: ", time.time()-t)
result_wire_number = current.matrix[1,2*precision-1]
print("the last bit of the result is in wire number = ", current.matrix[1,2*precision-1])
print (" the output is : ", wires[result_wire_number])
print("it is authenticated if output = " , keys_euc[result_wire_number][9])
print("it is not authenticated if output = " , keys_euc[result_wire_number][0])