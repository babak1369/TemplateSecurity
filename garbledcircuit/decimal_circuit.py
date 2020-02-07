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