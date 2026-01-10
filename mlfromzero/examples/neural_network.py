import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from mlfromzero.deep_learning.dense import Dense
from mlfromzero.deep_learning.activation_functions import ReLu
from mlfromzero.deep_learning.loss_functions import Activation_Softmax_Loss_CategoricalCrossentropy
from mlfromzero.deep_learning.optimizers import SGD


nnfs.init()

X, y = spiral_data(samples=100, classes=3)


dense1 = Dense(2, 64)

activation1 = ReLu()

dense2 = Dense(64, 3)

loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

optimizer = SGD(learning_rate=.85)

for epoch in range(10001):

    dense1.forward(X)

    activation1.forward(dense1.output)

    dense2.forward(activation1.output)

    loss = loss_activation.forward(dense2.output, y)

    predictions = np.argmax(loss_activation.output, axis=1)

    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)

    accuracy = np.mean(predictions==y)

    if not epoch % 100:
        print(  f'epoch: {epoch}, ' +
                f'acc: {accuracy:.3f}, ' +
                f'loss: {loss:.3f}')
    # Backward pass
    loss_activation.backward(loss_activation.output, y)

    dense2.backward(loss_activation.dinputs)

    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    # Update weights and biases
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)