import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data.csv')

X = dataset.iloc[:,2:32] # [all rows, col from index 2 to the last one excluding 'Unnamed: 32']
y = dataset.iloc[:,1] # [all rows, col one only which contains the classes of cancer]

from sklearn.preprocessing import LabelEncoder

print("Before encoding: ")
print(y[100:110])

labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)

# print("\nAfter encoding: ")
# print(y[100:110])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, Adagrad, SGD, RMSprop
from tensorflow.keras.callbacks import TensorBoard

np.random.seed(42)

import time
NAME = f"{int(time.time())}"

model = Sequential()

model.add(Dense(16, activation= 'relu', input_dim= 30))
# model.add(Dropout(0.6))
model.add(Dense(8, activation= 'relu'))
# model.add(Dropout(0.2))
model.add(Dense(6, activation= 'relu'))
# model.add(Dropout(0.2))
model.add(Dense(1, activation= 'sigmoid'))

opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
# adagrad = Adagrad(lr=0.0001)
# sgd = SGD(lr=0.0001)
# rms = RMSprop(lr=0.0001)

model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])   # 'rmsprop'

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    batch_size = 10,
                    verbose=2,
                    epochs = 100,
                    callbacks=[tensorboard])

print(history.history.keys())
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()