from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import load_model

import numpy as np
import time


class LSTM_Network:
    def __init__(self, file):
        self.file = file
        self.model = object()
        self.fit_model = object()
        self.y_predicted = int()

    def define_model(self):
        self.model = Sequential()

        self.model.add(LSTM(32, return_sequences = True, batch_input_shape = (1,1,1), stateful = True))
        self.model.add(LSTM(32, return_sequences = True, stateful = True))
        self.model.add(Dense(1))
        self.model.compile(loss = 'mean_squared_error', optimizer = 'rmsprop')
        return self.model
    def load_model(self):
        self.model = load_model(self.file)
        return self.fit_model
    def save_model(self):
        self.model.save(self.file)

    def train(self, x, y):
        #Reshape Data
        x = np.reshape(x, [-1,1,1])
        y = np.reshape(y, [-1,1,1])
        #Fit Data to model
        try:
            self.fit_model = self.model.fit(x, y, epochs = 1, batch_size = 1, verbose = 2)
            self.save_model()
            return self.fit_model
        except Exception as e:
            print(e)
            print("No Training Data Yet")
    def predict(self, x_predict):
        try:
            x_predict = np.reshape(x_predict, [-1,1,1])
            self.y_predict = self.model.predict(x_predict)
            self.y_predict = self.y_predict[-1]
        except:
            self.y_predict = 0
        return self.y_predict
