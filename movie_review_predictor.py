import tensorflow as td
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

# https://keras.io/preprocessing/text/
# num_words: the maximum number of words to keep (most frequent words)
(train_data, train_labels), (test_data,
                             test_labels) = data.load_data(num_words=10000)

word_index = data.get_word_index()
word_index = {k: (v+3) for k, v in word_index.items()}
# v+3: have some custom parameters
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3

reverse_word_index = dict([(value, key)
                           for (key, value) in word_index.items()])

# Preprocessing the data to the same length
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data, value=word_index['<PAD>'], padding='post', maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(
    test_data, value=word_index['<PAD>'], padding='post', maxlen=250)

model = keras.Sequential()
# Embedding layer: give neurons senmantic similarity
model.add(keras.layers.Embedding(10000, 16))
# GlobalAveragePooling1D layer: demension deduction
model.add(keras.layers.GlobalAveragePooling1D())
# Hidden layer
model.add(keras.layers.Dense(16, activation='relu'))
# Output neuron
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# validation set is to adjust the hyperparameter
x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

# batch_size: how many entites will be loaded into the memory at a time
fitModel = model.fit(x_train, y_train, epochs=40,
                     batch_size=512, validation_data=(x_val, y_val), verbose=1)

result = model.evaluate(test_data, test_labels)

# Save model
model.save('movie_review_predictor_model.h5')

# Import model
# model.keras.models.load_model('movie_review_predictor_model.h5')


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


if __name__ == '__main__':
    # print(decode_review(test_data[0]))
    print(result)
