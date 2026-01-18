import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from nnfs.datasets import sine_data

from mlfromzero.deep_learning.layer import Dense, Dropout
from mlfromzero.deep_learning.activation_functions import ReLU, Linear, Sigmoid, Softmax
from mlfromzero.deep_learning.loss_functions import Activation_Softmax_Loss_CategoricalCrossentropy, MeanSquaredError, BinaryCrossentropy, CategoricalCrossentropy
from mlfromzero.deep_learning.optimizers import SGD, Adam
from mlfromzero.deep_learning.model import Model
from mlfromzero.deep_learning.accuracy import Accuracy, Regression, Categorical

nnfs.init()

X, y = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

model = Model()

model.add(
    Dense(
        2,
        512,
        weight_regularizer_l2=5e-4,
        bias_regularizer_l2=5e-4
    )
)
model.add(ReLU())
model.add(Dropout(0.1))
model.add(Dense(512, 3))
model.add(Softmax())

model.set(
    loss=CategoricalCrossentropy(),
    optimizer=Adam(learning_rate=0.05, decay=5e-5),
    accuracy=Categorical()
)

model.finalize()

model.train(
    X,
    y,
    validation_data=(X_test, y_test),
    epochs=100,
    print_every=100
)

model.evaluate(X_test, y_test)