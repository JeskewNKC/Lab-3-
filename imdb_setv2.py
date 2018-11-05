# MLP for the IMDB problem
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.preprocessing import sequence
from keras.optimizers import Adam, Adagrad, SGD, RMSprop
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

# fix random seed for reproducibility
np.random.seed(42)

# load the dataset but only keep the top n words, zero the rest
top_words = 4000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

import time
NAME = f"{int(time.time())}"

# convolutional neural network model
model = Sequential()
model.add(Embedding(top_words, 32, input_length=max_words))
model.add(Conv1D(filters=32, kernel_size=3,  activation='relu'))
model.add(MaxPooling1D(pool_size=2, padding='same'))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

opt = Adam(lr=0.0001)
# sgd = SGD(lr=0.0001)
# opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
adagrad = Adagrad(lr=0.0001)
# rms = RMSprop(lr=0.0001)
model.compile(loss='binary_crossentropy',
              optimizer=adagrad,
              metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
print(model.summary())



# Fit the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=512,
    verbose=2,
    callbacks=[tensorboard])
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()