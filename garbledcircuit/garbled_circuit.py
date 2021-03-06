import numpy as np
from .utils import wire,matricize
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
    def forced_keys_circuit_garbling(self,forced_keys,numbers):
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
        all_keys[0:numbers] = forced_keys
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