import numpy as np
class DecimalCircuit:

    def __init__(self):
        self.add_counter = 0
        self.mult_counter = 0


    def addition_gate(self,a,b):
        if isinstance(a, (np.ndarray)):
            self.add_counter = self.add_counter + a.size
        else:
            self.add_counter = self.add_counter + 1
        o0 = (a+b) % 10
        o1 = np.floor(((a+b)/10))
        return [o0, o1]

    def mult_gate(self,a,b):

        if isinstance(a, (np.ndarray)):
            self.mult_counter = self.mult_counter + a.size
        else:
            self.mult_counter = self.mult_counter + 1
        o0 = (a*b) % 10
        o1 = np.floor(a*b/10)
        return [o0, o1]

    def addition_gate_1_2(self,a,b,c):
        o0,o1 = self.addition_gate(a,b)
        o2,o3 = self.addition_gate(o1,c)
        return [o0,o2]

    #a and b are digits
    def addition_circuit(self,a,b,number_of_digits):
        o0,o1 = self.addition_gate(a,b)
        output=[]
        temp = 0
        for i in range(number_of_digits):
            out1,out2 = self.addition_gate_1_2(temp,o0[i],o1[i])
            temp = out2
            output.append(int(out1))
        return output

    def multiplication_circuit(self,a,b,number_of_digits):
        b1 = np.reshape(np.repeat(b,number_of_digits),(number_of_digits,number_of_digits))
        a1 = np.reshape(np.tile(a,number_of_digits),(number_of_digits,number_of_digits))

        mults0, mults1 = self.mult_gate(a1,b1)
        final = []

        for i in range(number_of_digits):
            res = []
            carry = 0
            for k in range(i):
                res.append(0)
            for j in range(number_of_digits):
                o0,o1 = self.addition_gate_1_2(carry, mults0[i,j], mults1[i,j])
                res.append(o0)
                carry = o1
            res.append(carry)
            final.append(res)
        m = len(final[-1])
        for i in final:
            i.extend([0]*(m-len(i)))
        temp = 0
        for i in range(number_of_digits):
            temp = self.addition_circuit(np.array(final[i]),temp,m)
        return temp
dc = DecimalCircuit()

a = np.array([2,4,2 ,4,9,2,4,2 ,4,9,2,4,2 ,4,9,2,4,2 ,4,9])
b = np.array([2,4,2 ,4,9,2,4,2 ,4,9,2,4,2 ,4,9,2,4,2 ,4,9])
# a = np.array([2,4,2 ,4,9,2,4,2  ])
# b = np.array([2,4,2 ,4,9,2,4,2  ])
import time
t = time.time()
c = dc.multiplication_circuit(a,b,20)
# c = dc.addition_circuit(a,b,8)
print(time.time()-t)
t = time.time()
94242942429424294242**2

print(time.time()-t)
print(c)
def matricize(v,direction=True):

    s =  v.size
    if direction:
        return np.reshape(v,(s,1))
    else:
        return np.reshape(v,(1,s))
class wire:
    def __init__(self,matrix):
        self.matrix= matrix


class GC:

    def __init__(self,security = 100):
        self.add_counter = 0
        self.mult_counter = 0
        self.wire_counter = 0
        self.circuit=  np.empty((0,5), int)
        self.zerowirecounter = -1
        self.security = security
    def addition_gate(self, a, b):
        self.add_counter = self.add_counter + a.matrix.shape[1]

        o0 = (a.matrix[0] + b.matrix[0]) % 10
        o0_number = np.arange(a.matrix.shape[1])+self.wire_counter
        self.wire_counter = self.wire_counter+a.matrix.shape[1]
        o1 = np.floor(((a.matrix[0] + b.matrix[0]) / 10))
        o1_number = np.arange(a.matrix.shape[1]) + self.wire_counter
        self.wire_counter = self.wire_counter + a.matrix.shape[1]
        out0 = wire(np.vstack((o0,o0_number)).astype(np.int32))
        out1 = wire(np.vstack((o1, o1_number)).astype(np.int32))
        gate_type = np.zeros((a.matrix.shape[1]))
        circuit = np.transpose(np.vstack((a.matrix[1],b.matrix[1],o0_number,o1_number,gate_type)))
        self.circuit = np.vstack((self.circuit,circuit)).astype(np.int32)
        return out0,out1

    def mult_gate(self, a, b):
        self.mult_counter = self.mult_counter + a.matrix.shape[1]

        o0 = (a.matrix[0] * b.matrix[0]) % 10
        o0_number = np.arange(a.matrix.shape[1])+self.wire_counter
        self.wire_counter = self.wire_counter+a.matrix.shape[1]
        o1 = np.floor(((a.matrix[0] * b.matrix[0]) / 10))
        o1_number = np.arange(a.matrix.shape[1]) + self.wire_counter
        self.wire_counter = self.wire_counter + a.matrix.shape[1]
        out0 = wire(np.vstack((o0,o0_number)).astype(np.int32))
        out1 = wire(np.vstack((o1, o1_number)).astype(np.int32))

        gate_type = np.ones((a.matrix.shape[1]))
        circuit = np.transpose(np.vstack((a.matrix[1],b.matrix[1],o0_number,o1_number,gate_type)))
        self.circuit = np.vstack((self.circuit,circuit)).astype(np.int32)
        return out0,out1
    def addition_gate_1_2(self,a,b,c):
        o0,o1 = self.addition_gate(a,b)
        o2,o3 = self.addition_gate(o1,c)
        return [o0,o2]
    def addition_circuit(self,a,b,number_of_digits):
        o0,o1 = self.addition_gate(a,b)
        output=  np.empty((2,0), int)
        output = np.hstack((output, matricize(o0.matrix[:,0])))
        temp = wire(matricize(o1.matrix[:,0]))
        for i in range(1,number_of_digits):
            in1 = wire(matricize(o0.matrix[:,i]))
            in2 = wire(matricize(o1.matrix[:,i]))
            out1,out2 = self.addition_gate_1_2(temp,in1,in2)
            temp = out2
            output = np.hstack((output,out1.matrix))


        return wire(output.astype(np.int32))
    def multiplication_circuit(self,a,b,number_of_digits, output_number_of_digits):
        b1_value =  np.repeat(b.matrix[0],number_of_digits)
        b1_wire_number =   np.repeat(b.matrix[1],number_of_digits)
        a1_value =  np.tile(a.matrix[0],number_of_digits)
        a1_wire_number = np.tile(a.matrix[1],number_of_digits)
        b1 = wire(np.vstack((b1_value,b1_wire_number)))
        a1 = wire(np.vstack((a1_value,a1_wire_number)))
        mults0, mults1 = self.mult_gate(a1,b1)
        mults0_t = np.transpose(mults0.matrix)
        mults1_t = np.transpose(mults1.matrix)
        mults0 = np.reshape(mults0_t,(number_of_digits,number_of_digits,2))
        mults1 = np.reshape(mults1_t, (number_of_digits, number_of_digits, 2))
        final = []

        for i in range(number_of_digits):
            res = np.empty((2,0))
            carry = wire(np.array([[0],[-1]]))
            for k in range(i):
                res = np.hstack((res,np.array([[0],[self.zerowirecounter]])))
                self.zerowirecounter = self.zerowirecounter -1
            for j in range(number_of_digits):
                o0,o1 = self.addition_gate_1_2((carry), wire(matricize(mults0[i,j])), wire(matricize(mults1[i,j])))
                res = np.hstack((res,o0.matrix))
                carry = o1
            res = np.hstack((res,o1.matrix))
            final.append(res)
        for i in range (len(final)):
            for j in range(output_number_of_digits-final[i].shape[1]):
                final[i] = np.hstack((final[i],np.array([[0],[self.zerowirecounter]])))
                self.zerowirecounter = self.zerowirecounter -1

        temp = wire(final[0][:,0:output_number_of_digits])
        for i in range(1,number_of_digits):
            numb1 =  wire(final[i][:,0:output_number_of_digits])
            temp = self.addition_circuit(numb1,temp,output_number_of_digits)
        return temp
    def table_encryption(self,outputlist,inputlist):
        res = np.zeros((1,2,self.security+1))
        for i in range(2):
            res[0,i,0:self.security] = ((outputlist[i][0:self.security]+inputlist[0][0:self.security] +inputlist[1][0:self.security]) %2).reshape((1,self.security))
            res[0,i,self.security] = (outputlist[i][self.security]-inputlist[0][self.security]-inputlist[1][self.security] )% 10
        return res
    def add_garbled(self,wires,keys):
        index = wires
        encrypted_table = np.zeros((100,2,self.security+1),dtype=np.int32)
        for i in range(10):
            for j in range(10):
                row = 10 * keys[index[0]][i ,self.security ] + keys[index[1]][j ,self.security]
                o0 = int((i + j) % 10)
                o1 = int(np.floor(((i + j) / 10)))
                encrypted_table[row,:] = self.table_encryption([keys[index[2]][o0],keys[index[3]][o1]],
                                                                 [keys[index[0]][i],keys[index[1]][j]])
        return encrypted_table
    def mult_garbled(self,wires,keys):
        index = wires
        encrypted_table = np.zeros((100,2,self.security+1),dtype=np.int32)
        for i in range(10):
            for j in range(10):
                row = 10 * keys[index[0]][i ,self.security ] + keys[index[1]][j ,self.security]
                o0 = int((i * j) % 10)
                o1 = int(np.floor(((i * j) / 10)))
                encrypted_table[row,:] = self.table_encryption([keys[index[2]][o0],keys[index[3]][o1]],
                                                                 [keys[index[0]][i],keys[index[1]][j]])
        return encrypted_table

    def circuit_garbling(self):
        max_ = self.circuit.max()
        min_ = self.zerowirecounter
        no_wires = max_ - min_
        all_keys = []
        for i in range (no_wires):
            keys = np.random.randint(0,2,(10,self.security))
            perm = np.random.permutation(10).reshape((10,1))
            keys = np.hstack((keys,perm)).astype(np.int32)
            all_keys.append(keys)
        encrypted_gates=[]
        for i in range (self.circuit.shape[0]):
            if self.circuit[i,4] == 0:
                encrypted_gates.append(self.add_garbled(self.circuit[i,0:4],all_keys))
            else:
                encrypted_gates.append(self.mult_garbled(self.circuit[i, 0:4], all_keys))

        return encrypted_gates,all_keys
    def degarbling(self,encrypted_table,given_keys):
        for i in range(self.circuit.shape[0]):
            key1 = given_keys[self.circuit[i,0]]
            key2 = given_keys[self.circuit[i,1]]
            row = int(10*key1[self.security]+key2[self.security])
            n_rows1 = (encrypted_table[i][row,0,self.security]+key1[self.security]+key2[self.security])%10
            n_rows2 = (encrypted_table[i][row, 1, self.security] + key1[self.security] + key2[self.security]) % 10
            k_out1 = ((encrypted_table[i][row,0,0:self.security]+key1[0:self.security]+key2[0:self.security])%2)
            k_out2 = ((encrypted_table[i][row, 1, 0:self.security] + key1[0:self.security] + key2[0:self.security]) % 2)
            k_out1 = np.append(k_out1,n_rows1)
            k_out2 = np.append(k_out2,n_rows2)
            given_keys[self.circuit[i,2]]= k_out1
            given_keys[self.circuit[i, 3]]= k_out2
        return given_keys
    def group_degarbling(self,encrypted_table,given_keys):
        batch_size = given_keys.shape[0]
        for i in range(self.circuit.shape[0]):
            key1 = given_keys[:,self.circuit[i,0]]
            key2 = given_keys[:,self.circuit[i,1]]
            row =  10*key1[:,self.security]+key2[:,self.security]
            n_rows1 = (encrypted_table[i][row,0,self.security]+key1[:,self.security]+key2[:,self.security])%10
            n_rows2 = (encrypted_table[i][row, 1, self.security] + key1[:,self.security] + key2[:,self.security]) % 10
            k_out1 = ((encrypted_table[i][row,0,0:self.security]+key1[:,0:self.security]+key2[:,0:self.security])%2)
            k_out2 = ((encrypted_table[i][row, 1, 0:self.security] + key1[:,0:self.security] + key2[:,0:self.security]) % 2)
            k_out1 = np.hstack((k_out1,n_rows1.reshape((batch_size,1))))
            k_out2 = np.hstack((k_out2,n_rows2.reshape((batch_size,1))))
            given_keys[:,self.circuit[i,2]]= k_out1
            given_keys[:,self.circuit[i, 3]]= k_out2
        return given_keys


import math
class TemplateSecurity:
    """
    template = numpy array
    """
    def __init__(self, template, precision,threshold):
        self.template = template
        self.precision = precision
        self.wire_counter = 0
        self.dimension = template.size
        self.square_sum = np.sum(template*template)
        self.threshold = 10**(2*precision) - (threshold**2)

    def preprocessing(self, output ,precision=None):
        elements = []
        if precision is None:
            precision = self.precision
        for i in range(output.size):
            digits = np.zeros((2, precision))
            for j in range(len(str(output[i]))):
                digits[0,j]= output[i] % 10
                output[i]=int(output[i]//10)
            digits[1] = np.arange(precision)+ self.wire_counter
            self.wire_counter = self.wire_counter + precision
            elements.append(digits.astype(np.int64))
        return elements


    def euclidean_distance(self):
        gc_euc = GC()

        A= self.preprocessing(np.ones(self.template.shape))
        B = self.preprocessing(self.template)
        threshold = self.preprocessing(np.array([self.threshold]), 2*self.precision)[0]
        square_sum_template =self.preprocessing(np.array([self.square_sum]), 2*self.precision)[0]
        square_sum_query =self.preprocessing(np.array([2]), 2*self.precision)[0]
        constant_negative_2 = self.preprocessing(np.array([10**20-2]),2*self.precision)[0]
        wires_list_a=[]
        wires_list_b=[]
        mult_list = []
        gc_euc.wire_counter = self.wire_counter
        for i in range(self.dimension):
            wires_list_a.append(wire(A[i]))
            wires_list_b.append(wire(B[i]))
            mult_list.append(gc_euc.multiplication_circuit(wire(A[i]),wire(B[i]),self.precision,2*self.precision))
        current = mult_list[0]
        for i in range(1,len(mult_list)):
            current = gc_euc.addition_circuit(current,mult_list[i],2*self.precision)
        current = gc_euc.multiplication_circuit(wire(constant_negative_2),current,2*self.precision,2*self.precision)
        current = gc_euc.addition_circuit(current,wire(square_sum_query),2*self.precision)
        current = gc_euc.addition_circuit(current, wire(square_sum_template), 2*self.precision)
        current = gc_euc.addition_circuit(current,wire(threshold),2*self.precision)
        return [gc_euc,current, A,B,threshold,square_sum_template,square_sum_query,constant_negative_2]

    def euclidean_distance_setup(self):
        gc_euc, current, A, B, threshold, square_sum_template, square_sum_query,constant_negative_2 = self.euclidean_distance()

        et, keys = gc_euc.circuit_garbling()
        max_ = gc_euc.circuit.max()
        min_ = gc_euc.zerowirecounter
        no_wires = max_ - min_
        wires = np.zeros((no_wires, gc_euc.security + 1), dtype=np.int32)
        for i in range(1, abs(min_)):
            wires[-i] = keys[-i][0]
        for i in range (len(B)):
            for j in range(self.precision):
                wires[B[i][1,j]]=keys[B[i][1,j]][B[i][0,j]]
        for i in range (2*self.precision):
            wires[threshold[1,i]] = keys[threshold[1,i]][threshold[0,i]]
            wires[square_sum_template[1, i]] = keys[square_sum_template[1, i]][square_sum_template[0, i]]
            wires[constant_negative_2[1, i]] = keys[constant_negative_2[1, i]][constant_negative_2[0, i]]

        return wires,et,keys,B,square_sum_query,current,gc_euc
    def prepare_query(self,query,square_sum,wires,keys):
        """"
        keys are precision * d for input
        + 2*precision for square
        """
        self.wire_counter = 0
        value_square = np.sum(query * query)
        A = self.preprocessing(query,self.precision)
        square_sum_query = self.preprocessing(np.array([value_square]), 2 * self.precision)[0]
        for i in range(len(A)):
            for j in range(self.precision):
                wires[A[i][1, j]] = keys[A[i][1, j]][A[i][0, j]]
        for i in range (2*self.precision):
            #TODO: KEY size is 50.. needs generalization
            wires[square_sum[1,i]] = keys[i+len(A)*self.precision][square_sum_query[0,i]]

        return wires,A

import time
# prec = 10
# no = 4000
# A = np.array([[2,0,0,0,0,0,0,0,0,0],[0,1,2,3,4,5,6,7,8,9]])
# #B = np.array([np.random.randint(0,9,(prec)),np.arange(prec)+prec])
# B = np.array([[1,1,0,0,0,0,0,0,0,0],np.arange(10)+10])
#
# C = np.array([[8,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],np.arange(20)])
# #
# #C = np.array([[0,0,1,1,0],[10,11,12,13,14]])
# a = wire(A)
# b = wire(B)
# #c = wire(C)
# #c= wire(C)
# gc = GC()
# gc.wire_counter = 40
# #m = gc.addition_circuit(a,b,prec)
# m = gc.multiplication_circuit(a,b,prec,2*prec)
# #m1 = gc.multiplication_circuit(m,c,2*prec,2*prec)
# # m2= gc.addition_circuit(m1,c,2*prec)
# et,keys = gc.circuit_garbling()
#
# max_ = gc.circuit.max()
# min_ = gc.zerowirecounter
# no_wires = max_ - min_
# wires=np.zeros((no_wires,gc.security+1),dtype=np.int32)
# for i in range(prec):
#     wires[i] = keys[i][A[0,i]]
# for i in range(prec,2*prec):
#     wires[i] = keys[i][B[0,i-prec]]
# for i in range(1,abs(min_)):
#     wires[-i] = keys[-i][0]
# gwires = np.expand_dims(wires, axis=0)
# gwires = np.tile(gwires,(no,1,1))
# # t= time.time()
# wi= gc.degarbling(et,wires)
# # print( no*(time.time()-t))
#
# t = time.time()
# wi = gc.group_degarbling(et,gwires)
# print(time.time()-t)
A = np.array([0,23,24])
dimension = 3
precision = 10
ts = TemplateSecurity(A,precision,4)
wires,et,keys,A,square_sum_query,current,gc_euc =  ts.euclidean_distance_setup()
keys1 = keys[0:dimension*precision]+keys[square_sum_query[1,0]:square_sum_query[1,0]+2*precision]
query = np.array([0,25,23])


wires,B = ts.prepare_query(query,square_sum_query,wires,keys1)

t= time.time()
wi= gc_euc.degarbling(et,wires)
print(time.time()-t)
