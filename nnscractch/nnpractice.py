import numpy as np
import sys
import matplotlib as plt

inputs = [1 , 2, 3, 2.5]
weight1 = [0.2, 0.8, -0.5, 1.0]
weight2 = [0.5, -0.91, 0.26, -0.5]
weight3 = [-0.26, -0.26, 0.17, 0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5

output = inputs[0] * weight[0] + inputs[1] * weight[1] + inputs[2] * weight[2] + inputs[3] * weight[3] + bias
print(output)