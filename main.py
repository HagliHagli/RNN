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
attribute = 'Open'
input_data = data[attribute][:timestep * LSTM_input]

#processing
output = S.Process(input_data)
print(output)