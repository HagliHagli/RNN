import numpy as np
import pandas as pd

class Dense :
    # class members
    # unit_num -> number of units
    # mode -> what mode is used, 1 for ReLU, 2 for sigmoid
    # input = [y, x, z] -> the input matrix
    #
    # class methods
    # EnterInput (input) -> enter a matrix to the input class member
    # Process -> apply the categorizing process
    def __init__(self) :
        print('Dense layer creation.')

        self.unit_num = int(input('Input the amount of units.'))
        
        self.mode = int(input('Input the mode of the phase.\nType "1" for ReLU. Type "2" for sigmoid.'))
        while (self.mode != 1 and self.mode != 2) :
            print('Please enter either 1 (ReLU mode) or 2 (sigmoid mode).')
            self.mode = int(input())
        self.first = True
    
    def __init__(self, unit_num, mode) :
      self.unit_num = unit_num
      
      self.mode = mode
      self.first = True
            
    def EnterInput(self, matrix) :
        self.input = matrix
        if self.first :
            self.weight = np.random.rand(self.input.flatten().shape[0], self.unit_num) # weight's shape is (input.flatten().shape, node_num)
            self.bias = np.random.rand(self.unit_num)
            self.first = False
    
    def Summary(self) :
      if (self.first) :
        return 'Please enter an input first, dense layer.'
      f = lambda val : 'ReLU' if val==1 else 'Sigmoid'
      data = [('Number of units', self.unit_num),
              ('Mode', f(self.mode)),
              ('Weight', self.weight),
              ('Bias', self.bias)]
      return pd.DataFrame(data, columns = ['Attributes', 'value'])
    
    def Process(self) : # input's shape is the same as previous functions
        input_ = self.input.flatten()
        #print(input_.shape)
        output = np.ndarray(shape=(self.weight.shape[1]))
        #print(output.shape)
        for i in range(self.weight.shape[1]) :
            num = 0
            for j in range(self.weight.shape[0]) :
                #print(j)
                num = num + input_[j] * self.weight[j][i]
            if self.mode == 1 :
                if num < 0 :
                    output[i] = 0
                else :
                    output[i] = num
            elif self.mode == 2 :
                output[i] = 1/(1 + np.exp(-num))
            output[i] = output[i] + self.bias[i]
        self.output = output
        return output
    
    def ShowSoftmax(self) :
        sum = 0
        for i in self.output :
            sum = i + sum
        sfm = []
        for i in self.output :
            sfm.append(i/sum)
        return np.array(sfm)