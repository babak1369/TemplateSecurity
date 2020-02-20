import garbledcircuit as gc
import numpy as np
dc = gc.DecimalCircuit()
"""
This Example shows how the addition and multiplication decimal circuit work.


"""

""""
Multiplication
"""
precision = 20
a = np.array([2,4,2 ,4,9,2,4,2 ,4,9,2,4,2 ,4,9,2,4,2 ,4,9])
b = np.array([2,4,2 ,4,9,2,4,2 ,4,9,2,4,2 ,4,9,2,4,2 ,4,9])
c = dc.multiplication_circuit(a,b,precision)
#verification
print("Test_Multiplication: ")
print("output: " ,c)
print("expected_result: " , 94242942429424294242**2)

""""
addition
"""
precision = 20
a = np.array([2,4,2 ,4,9,2,4,2 ,4,9,2,4,2 ,4,9,2,4,2 ,4,9])
b = np.array([2,4,2 ,4,9,2,4,2 ,4,9,2,4,2 ,4,9,2,4,2 ,4,9])
c = dc.addition_circuit(a,b,precision)
#verification
print("Test_Addition: ")
print("output: " ,c)
print("expected_result: " ,94242942429424294242*2)
