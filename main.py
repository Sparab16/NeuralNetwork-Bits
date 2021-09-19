import pandas as pd

from NeuralNetworkBits import Neural

neural_obj = Neural()

bits = pd.DataFrame({'x1':(0,0,0,0,1,1,1,1), 'x2':(0,0,1,1,0,0,1,1), 'x3':(0,1,0,1,0,1,0,1), 'y':(0,1,2,3,4,5,6,7)})
print(bits)

inputs = bits[['x1', 'x2', 'x3']]
targets = bits['y']

final_weights = neural_obj.train(inputs, targets, 0.1, 20)

# Testing of the Neural Network
print(neural_obj.activation([[1,1,0,-1]], final_weights))
print(neural_obj.activation([[0,0,0,-1]], final_weights))
print(neural_obj.activation([[0,1,1,-1]], final_weights))
print(neural_obj.activation([[1,1,1,-1]], final_weights))