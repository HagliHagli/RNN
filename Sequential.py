#Sequential
class Sequential :
  def __init__(self, list_of_layers) :
    self.layers = list_of_layers
  
  def Summary(self) :
    data = []
    for i in self.layers :
      data = data + [i.Summary()]
    return data
      

  def Process(self, input_val) :
    for i in self.layers :
      i.EnterInput(input_val)
      input_val = i.Process()
    return input_val