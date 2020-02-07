import numpy as np
from .garbled_circuit import GC
from .utils import wire
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

    def parallel_euc_distance(self):
        gc = GC()
        prec = self.precision
        all_wires = 2 * self.precision * self.dimension + 2 * self.precision * 4
        no_wires_for_mult =  2 * self.precision * self.dimension
        gc.wire_counter = 2*self.precision
        A = np.array([np.random.randint(0,9,(prec)),np.arange(prec)])
        B = np.array([np.random.randint(0,9,(prec)),np.arange(prec)+prec])
        self.wire_counter = no_wires_for_mult
        threshold = self.preprocessing(np.array([self.threshold]), 2*self.precision)[0]
        self.wire_counter = no_wires_for_mult+2*self.precision
        square_sum_template =self.preprocessing(np.array([self.square_sum]), 2*self.precision)[0]
        self.wire_counter = no_wires_for_mult + 4 * self.precision
        square_sum_query =self.preprocessing(np.array([2]), 2*self.precision)[0]
        self.wire_counter = no_wires_for_mult + 6 * self.precision
        constant_negative_2 = self.preprocessing(np.array([10**20-2]),2*self.precision)[0]


        m = gc.multiplication_circuit(wire(A),wire(B),prec,2*prec)
        list_of_gc = self.dimension * [gc]
        gc_euc = GC()

        gc_euc.wire_counter = all_wires
        mult_result = np.zeros([2,2*prec*self.dimension])
        mult_result[1,:] = np.arange(2*self.dimension*prec)
        current = wire(mult_result[:,0:2*prec])
        for i in range(1,self.dimension):
            current = gc_euc.addition_circuit(current,wire(mult_result[:,i*2*prec:i*2*prec+2*prec ] ), 2*self.precision)
        current = gc_euc.multiplication_circuit(wire(constant_negative_2),current,2*self.precision,2*self.precision)
        current = gc_euc.addition_circuit(current,wire(square_sum_query),2*self.precision)
        current = gc_euc.addition_circuit(current, wire(square_sum_template), 2*self.precision)
        current = gc_euc.addition_circuit(current,wire(threshold),2*self.precision)
        return current, list_of_gc,gc_euc,m,threshold,square_sum_template,square_sum_query,constant_negative_2

    def parallel_euc_setup(self):

        """
        first run self.paralle.euc
        then we garble all the mult_circuits
        then we create parallel_garbling function and after random generation of all keys
        we replace wire 0-20 with keys corresponding to the output of gc for mult 1
        ...
        and
        :return:

        """
        current, list_of_gc,gc_euc,m,threshold,square_sum_template,square_sum_query,constant_negative_2 = self.parallel_euc_distance()
        et_list = []
        keys_list = []
        keys_list_mult = []
        counter = 1
        for garbled_circuit in list_of_gc:
            print(counter)
            counter = counter + 1
            et, keys = garbled_circuit.circuit_garbling()
            et_list.append(et)
            keys_list_mult.append(keys)
            for i in range (m.matrix.shape[1]):
                keys_list.append(keys[m.matrix[1,i]])
        et_euc,keys_euc = gc_euc.forced_keys_circuit_garbling(keys_list,2*self.precision*self.dimension)
        max_ = gc_euc.circuit.max()
        min_ = gc_euc.zerowirecounter
        no_wires = max_ - min_
        wires_euc = np.zeros((no_wires, gc_euc.security + 1), dtype=np.int32)
        for i in range(1, abs(min_)):
            wires_euc[-i] = keys_euc[-i][0]
        max_ = garbled_circuit.circuit.max()
        min_ = garbled_circuit.zerowirecounter
        mult_no_wire= max_ - min_

        wires_mult = np.zeros((self.dimension,mult_no_wire,list_of_gc[0].security+1),dtype=np.int32)
        for j in range (self.dimension):

            for i in range(1, abs(min_)):
                wires_mult[j,-i] = keys_list_mult[j][-i][0]
        for k in range(self.dimension):
            self.wire_counter = 0
            input_ = self.preprocessing(np.array([self.template[k]]),self.precision)[0]
            for i in range (self.precision):
                wires_mult[k,i,:] = keys_list_mult[k][i][input_[0,i]]
        for i in range (2*self.precision):
            wires_euc[threshold[1,i]] = keys_euc[threshold[1,i]][threshold[0,i]]
            wires_euc[square_sum_template[1, i]] = keys_euc[square_sum_template[1, i]][square_sum_template[0, i]]
            wires_euc[constant_negative_2[1, i]] = keys_euc[constant_negative_2[1, i]][constant_negative_2[0, i]]
        return wires_mult,wires_euc,et_list,et_euc,gc_euc,list_of_gc, keys_euc,keys_list_mult,square_sum_query,current,m

    def parallel_prepare_query(self, query,square_sum,wires_euc,wires_mult,available_keys_euc,available_keys_mult):
        self.wire_counter = self.precision
        value_square = np.sum(query * query)
        square_sum_query = self.preprocessing(np.array([value_square]), 2 * self.precision)[0]
        for i in range(2 * self.precision):
            wires_euc[square_sum[1, i]] = available_keys_euc[i][square_sum_query[0, i]]
        for k in range(self.dimension):
            self.wire_counter = self.precision
            input_ = self.preprocessing(np.array([query[k]]),self.precision)[0]
            for i in range (self.precision):
                wires_mult[k,self.precision+i,:] = available_keys_mult[k,i,input_[0,i]]
        return wires_euc,wires_mult

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

