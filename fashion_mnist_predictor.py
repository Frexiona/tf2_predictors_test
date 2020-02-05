import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# The number of epochs is a hyperparameter that defines the number times 
# that the learning algorithm will work through the entire training dataset
# epochs is not a constant
model.fit(train_images, train_labels, epochs=5)

# Parameter in the model predictot should be a LIST
prediction = model.predict(test_images)

# Evaluation
# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print('Tested acc:', test_acc)