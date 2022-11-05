from Dense import Dense
from LSTM import LSTM
from Sequential import Sequential

import pandas as pd

#making of the sequential model
timestep = 32
LSTM_input = 2
S = Sequential([LSTM(LSTM_input,3), Dense(1,1)])

#data preprocessing
data = pd.read_csv('./data/ETH-USD-Test.csv')
attribute = 'High'
input_data = data[attribute][:timestep * LSTM_input]
print('Atribut input : '+attribute)
print('Input : (5 teratas)')
print(input_data.head())

#processing
output = S.Process(input_data)
print('Output :')
print(output)