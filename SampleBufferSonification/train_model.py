from model_class import *
from data_save import loadData


data = loadData()


myModel = LSTM_Network('myModel.h5')

try:
    myModel.load_model()
except Exception as e:
    myModel.define_model()


myModel.train(data['x_time'],data['y_valence'])
