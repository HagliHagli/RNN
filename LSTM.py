import numpy as np
import pandas as pd

class LSTM :
    def __init__(self, input_size, hidden_size) :
        self.timestep = 32
        self.h = np.zeros(hidden_size)
        self.c = np.zeros(hidden_size)
        
        self.U_f = np.random.rand(hidden_size, input_size)
        self.W_f = np.random.rand(hidden_size, hidden_size)
        self.b_f = np.random.rand(hidden_size)
        
        self.U_i = np.random.rand(hidden_size, input_size)
        self.W_i = np.random.rand(hidden_size, hidden_size)
        self.b_i = np.random.rand(hidden_size)
        
        self.U_c_hat = np.random.rand(hidden_size, input_size)
        self.W_c_hat = np.random.rand(hidden_size, hidden_size)
        self.b_c_hat = np.random.rand(hidden_size)
        
        self.U_o = np.random.rand(hidden_size, input_size)
        self.W_o = np.random.rand(hidden_size, hidden_size)
        self.b_o = np.random.rand(hidden_size)

        self.input_size = input_size
        
    def EnterInput(self, val) : #total_size val = timestep * input_size
        self.input_val = val

    def Summary(self) :
      data = [('Timestep', self.timestep),
              ('h-1', self.h),
              ('c-1', self.c),
              ('U_f', self.U_f),
              ('W_f', self.W_f),
              ('b_f', self.b_f),
              ('U_i', self.U_i),
              ('W_i', self.W_i),
              ('b_i', self.b_i),
              ('U_c_hat', self.U_c_hat),
              ('b_c_hat', self.b_c_hat),
              ('U_o', self.U_o),
              ('W_o', self.W_o),
              ('b_o', self.b_o)]
      return pd.DataFrame(data, columns = ['Attributes', 'value'])
        
    def ProcessCell(self, cur_input) :
        forget = np.add(np.matmul(self.U_f, cur_input), np.matmul(self.W_f, self.h))
        forget = np.add(forget, self.b_f)
        forget = 1/(1 + np.exp(-forget))
        
        input_gate = np.add(np.matmul(self.U_i, cur_input), np.matmul(self.W_i, self.h))
        input_gate = np.add(input_gate, self.b_i)
        input_gate = 1/(1 + np.exp(-input_gate))
        
        c_hat = np.add(np.matmul(self.U_c_hat, cur_input), np.matmul(self.W_c_hat, self.h))
        c_hat = np.add(c_hat, self.b_c_hat)
        c_hat = np.tanh(c_hat)
        
        self.c = np.add(np.multiply(forget, self.c), np.multiply(input_gate, c_hat))
        
        output = np.add(np.matmul(self.U_o, cur_input), np.matmul(self.W_o, self.h))
        output = np.add(output, self.b_o)
        self.output = 1/(1 + np.exp(-output))
        
        self.h = np.multiply(self.output, np.tanh(self.c))
    
    def Process(self) : #process through time
        for i in range(self.input_size, self.timestep) :
            self.ProcessCell(self.input_val[i-self.input_size:i])
        return self.output