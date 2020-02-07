import numpy as np
from TemplateSecurity.garbledcircuit.utils import wire
from TemplateSecurity.garbledcircuit.garbled_circuit import GC
import time
"""
 2000 multiplications degarbling performance 
"""
prec = 10
no = 2000
A = np.array([[2,0,0,0,0,0,0,0,0,0],[0,1,2,3,4,5,6,7,8,9]])
#B = np.array([np.random.randint(0,9,(prec)),np.arange(prec)+prec])
B = np.array([[1,1,0,0,0,0,0,0,0,0],np.arange(10)+10])

C = np.array([[8,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],np.arange(20)])
#
#C = np.array([[0,0,1,1,0],[10,11,12,13,14]])
a = wire(A)
b = wire(B)
#c = wire(C)
#c= wire(C)
gc = GC()
gc.wire_counter = 40
#m = gc.addition_circuit(a,b,prec)
m = gc.multiplication_circuit(a,b,prec,2*prec)
#m1 = gc.multiplication_circuit(m,c,2*prec,2*prec)
# m2= gc.addition_circuit(m1,c,2*prec)
et,keys = gc.circuit_garbling()

max_ = gc.circuit.max()
min_ = gc.zerowirecounter
no_wires = max_ - min_
wires=np.zeros((no_wires,gc.security+1),dtype=np.int32)
for i in range(prec):
    wires[i] = keys[i][A[0,i]]
for i in range(prec,2*prec):
    wires[i] = keys[i][B[0,i-prec]]
for i in range(1,abs(min_)):
    wires[-i] = keys[-i][0]
gwires = np.expand_dims(wires, axis=0)
gwires = np.tile(gwires,(no,1,1))
# t= time.time()
wi= gc.degarbling(et,wires)
# print( no*(time.time()-t))

t = time.time()
wi = gc.group_degarbling(et,gwires)
print(time.time()-t)