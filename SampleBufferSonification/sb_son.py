# -*- coding: utf-8 -*- #?????
'Copyright (c) 2019 Jacob Colin Stucki III, All Rights Reserved.'
import time
import ctcsound #kernel
from ctcsound import * #The kernel
from random import randint # :)
import json
import nltk #word processing
from nltk import word_tokenize #tokens from strings
import os
import os.path #File containgurency cheks
import stat
import subprocess #For FIFO/File DNE Error handling
import sys
import logging #debugggg
from queue import Queue #Higher level FIFO
import uuid #We're using this as object pointers and shit, basically for congruency checks with the FIFO processing
import threading #Multithreading
import praw #Reddit API
import subprocess
import time
import random




from cPush import cPush
from Dictionary import Dictionary
from FIFO import FIFO
from File import File
from Neuron import Neuron
from Reddit import Reddit
from Transform import DataTransform
from Transform import Transformer

#Plotting Examples
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#scikit
import sklearn.linear_model
from sklearn.linear_model import LinearRegression

#keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import load_model
#pandas
import pandas as pd

from model_class import LSTM_Network
from data_save import *

import socket



jreddit = 'all'


logging.basicConfig(filename = 'Debug/debug_log_'+str(int(time.time()))+'.txt', level = logging.DEBUG)
print('####################### START #######################')




JSON_frame = { # REAL ONE IS UNSORTED, WE JUST HAVE IT NICE AND NEAT HERE BECUASE IT HELPS UNDERSTAND
    'emotion':{
        'happy':{ #init all zeros
            'generator':np.array([0]),
            'one_second_avg':np.array([0]),
            'sixty_second_avg':np.array([0])
        },
        'negative':{
            'generator':np.array([0]),
            'one_second_avg':np.array([0]),
            'sixty_second_avg':np.array([0])
        }
    },
    'uuid_old':str(),
    'raw':{
        'count':int(), #increment on "perception"
        'sum':int(), #Sum on "perception", because we don't want to do this at the mean level because we're outputing then and need to use it as a changing input
        'uuid':str(),
        'SSTime':float(),
        'subsphere':str(),
        'ID':str(),
        'author':str(),
        'body':str(),
        'other':str(),
        'systime':float(),
        'phase_angle':float()
    },
    'count':{
        'real':int(),
        'happy':int(),
        'negative':int(),
        'adjectives':int(),
        'adverbs':int(),
        'nouns':int(),
        'verbs':int(),
    },
    'valence':{
        'total':float(),
        'happy':float(),
        'negative':float()
    },
    'mean':{
        'activity':{ #this is the "average" (not really an average, but it changes every "update") number of samples
            'happy':{
                'event_count':int(),
                'event_count_old':int(),
                'event_activity':int()
            },
            'negative':{
                'event_count':int(),
                'event_count_old':int(),
                'event_activity':int()
            }
        },
        'emotion':{
            #'total':float(),
            'happy':float(),
            'negative':float()
        },
        'phase':float(),
        'delta':{
            'happy':{
                'old':float(),
                'delta':float(), #this one gets smaller and smaller because our mean approaches the daily(?) mean
                'delta_normal':float() #this one is normalized so that it is bounded to a float that is a multiple of 10^0 so that it is a meta-stagnant value
            },
            'negative':{
                'old':float(),
                'delta':float(),
                'delta_normal':float()
            }
        }
    },
    'legato':{
        'total':{
            'amp':float(),
            'fund_freq':float(),
            'fund_mod':float(),
            'mod_freq':float(),
            'mod_depth':float()
        },
        'happy':{
            'amp':float(),
            'fund_freq':float(),
            'fund_mod':float(),
            'mod_freq':float(),
            'mod_depth':float()
        },
        'negative':{
            'amp':float(),
            'fund_freq':float(),
            'fund_mod':float(),
            'mod_freq':float(),
            'mod_depth':float()
        }
    }
}

orch_legato = """
;Reference: http://www.adp-gmbh.ch/csound/fm/index.html
;http://www.adp-gmbh.ch/csound/opcodes/foscili.html

sr = 44100 ; (Samples/Second) aka how often we are going to be sampling
kr = 4410 ; Sample Rate / KSmps aka How often are we going to be changing the control values
ksmps = 10 ;number of samples in buffer
nchnls = 1 ; number of output channels 2 = stereo
0dbfs = 12 ; max db value

instr 1
    ;istart = p2
    ;idur = p3

    ;kAmp init p4
    ;kFreq init p5
    ;kCarrier init p6
    ;kMod init p7
    ;kIndex init p8

    kAmp chnget "kAmp" ; Amplitude
    kFundamentalFreq chnget "kFundamentalFreq" ; Fundamental Frequency
    kFundamentalMod chnget "kFundamentalMod" ; Carrier Freq = fundamental * carrier
    kModFreq chnget "kModFreq" ; Modulating Freq
    kModDepth chnget "kModDepth" ; index * fundamental * mod = index * freq_mod

    a1 foscili kAmp, kFundamentalFreq, kFundamentalMod, kModFreq, kModDepth

    out a1
endin
"""

dataTransformer = DataTransform


#Dict FILE Loads
FILE_realWords = File('Dictionaries/lotsofwords.txt')
FILE_happyWords = File('Dictionaries/happywords.txt')
FILE_negativeWords = File('Dictionaries/negativewords.txt')
FILE_adjectives = File('Dictionaries/adjectives.txt')
FILE_adverbs = File('Dictionaries/adverbs.txt')
FILE_nouns = File('Dictionaries/nouns.txt')
FILE_verbs = File('Dictionaries/verbs.txt')

#Log File Loads
reddit_Raw_Log = File('logs/reddit_Raw_Log.txt')
meta_raw_log = File('logs/meta_raw_log.txt')
valence_meta_log = File('logs/valence_meta_log.txt')
mean_log = File('logs/mean_log.txt')

#Dictionaries Load
DICT_real = Dictionary(name = 'real', type = 'basic_file', fileList = FILE_realWords.fileToList())
DICT_happy = Dictionary(name = 'happy', type = 'basic_file', fileList = FILE_happyWords.fileToList())
DICT_negative = Dictionary(name = 'negative', type = 'basic_file', fileList = FILE_negativeWords.fileToList())
DICT_adjectives = Dictionary(name = 'adjectives', type = 'basic_file', fileList = FILE_adjectives.fileToList())
DICT_adverbs = Dictionary(name = 'adverbs', type = 'basic_file', fileList = FILE_adverbs.fileToList())
DICT_nouns = Dictionary(name = 'nouns', type = 'basic_file', fileList = FILE_nouns.fileToList())
DICT_verbs = Dictionary(name = 'verbs', type = 'basic_file', fileList = FILE_verbs.fileToList())

#List of Dictionary Objects
list_DICT = [] #Order does not determine metadata, because we've switched to json
list_DICT.append(DICT_real)
list_DICT.append(DICT_happy)
list_DICT.append(DICT_negative)
list_DICT.append(DICT_adjectives)
list_DICT.append(DICT_adverbs)
list_DICT.append(DICT_nouns)
list_DICT.append(DICT_verbs)

#flags
stopFlag = threading.Event()

#Queues
QUEUE_Reddit_Raw = Queue()
QUEUE_Reddit_Meta = Queue()
QUEUE_valence_meta = Queue()
QUEUE_mean = Queue()

Legato_Happy = ctcsound.Csound()
Legato_Negative = ctcsound.Csound()

Legato_Happy.setOption("-odac")
Legato_Negative.setOption("-odac")
Legato_Happy.compileOrc(orch_legato)
Legato_Negative.compileOrc(orch_legato)


reddit_Stream_askreddit = Reddit(
                         name = 'Reddit'
                        ,flag = stopFlag
                        ,transform = dataTransformer
                        ,file = reddit_Raw_Log
                        ,fifo = QUEUE_Reddit_Raw
                        ,subreddit = jreddit
                        ,frame_obj = JSON_frame #only thing we're changing here is the count
                        )

Meta_Method_Obj = Neuron(
                        name = 'meta'
                        ,flag = stopFlag
                        ,transformObj = dataTransformer
                        ,file_output = meta_raw_log
                        ,referenceData = list_DICT
                        ,fifo_input = QUEUE_Reddit_Raw
                        ,fifo_out1 = QUEUE_Reddit_Meta
                        ,dto_Method = dataTransformer.dictListCompare
                        ,frame_obj = JSON_frame
                        )

Valence_Method_Obj = Neuron(
                        name = 'valence'
                        ,flag = stopFlag
                        ,transformObj = dataTransformer
                        ,file_output = valence_meta_log
                        ,referenceData = None
                        ,fifo_input = QUEUE_Reddit_Meta
                        ,fifo_out1 = QUEUE_valence_meta
                        ,dto_Method = dataTransformer.normalizeEmotion
                        ,frame_obj = JSON_frame
                        )


Rolling_Window_Obj = Transformer(frame_obj = JSON_frame, dto_method = dataTransformer.rolling_window)

Stats_Obj = Transformer(frame_obj = JSON_frame, dto_method = dataTransformer.showStats)

Legato_Method_Obj = Neuron(
                        name = 'legato'
                        ,flag = stopFlag
                        ,frame_obj = JSON_frame
                        ,dto_Method = dataTransformer.legato_map
                        )


cPush_Happy = cPush(
                    name = 'cPush_Happy'
                    ,legatoObj = Legato_Happy
                    ,frame = JSON_frame
                    ,flag = stopFlag
                    ,emotion = 'happy'
                    )
cPush_Negative = cPush(
                    name = 'cPush_Negative'
                    ,legatoObj = Legato_Negative
                    ,frame = JSON_frame
                    ,flag = stopFlag
                    ,emotion = 'negative'
)


AskRedditThread = threading.Thread(target = reddit_Stream_askreddit.pullComment) #We can't have () on the end of .pullComment because it is a stagnant method.
MetaThread = threading.Thread(target = Meta_Method_Obj.io_FRAME)
ValenceThread = threading.Thread(target = Valence_Method_Obj.io_FRAME)
LegatoFrameThread = threading.Thread(target = Legato_Method_Obj.io_FRAME)
RollingWindowThread = threading.Thread(target = Rolling_Window_Obj.io)

cPushHappyThread = threading.Thread(target = cPush_Happy.legatoPush)
cPushNegativeThread = threading.Thread(target = cPush_Negative.legatoPush)

Happy_LegatoCThread = CsoundPerformanceThread(Legato_Happy.csound())
Negative_LegatoCThread = CsoundPerformanceThread(Legato_Negative.csound())

stats_Thread = threading.Thread(target = Stats_Obj.io)


AskRedditThread.setDaemon(True) #Set so async thread will not block main from closing, but we still check later for .join() just for good practice
MetaThread.setDaemon(True)
ValenceThread.setDaemon(True)
RollingWindowThread.setDaemon(True)

LegatoFrameThread.setDaemon(True)
cPushHappyThread.setDaemon(True)
cPushNegativeThread.setDaemon(True)

stats_Thread.setDaemon(True)

data = loadData()
x_datetime = []
for i in range (0, len(data['x_valence'])):
    datetime_object = pd.to_datetime(data['x_valence'][i])
    x_datetime.append(datetime_object)


##Plots
x_zero, y_zero = [], []
figure = plt.figure(1)
zero_line, = plt.plot_date(x_zero, y_zero, '-', color = 'black', label = "Zero Line")
def zero_line_animate(frame):
    x_zero.append(datetime.now())
    y_zero.append(0)
    zero_line.set_data(x_zero, y_zero)
    figure.gca().relim()
    figure.gca().autoscale_view()
    return zero_line



x_happy, y_happy = [],[]
x_negative, y_negative = [], []
figure = plt.figure(1)
plt.title('60 Second Rolling Average - Valence')

line_happy, = plt.plot_date(x_happy,y_happy,'-', color = 'orange', label = 'positive') #So I don't understand the comma here
line_negative, = plt.plot_date(x_negative, y_negative, '-', color = 'blue', label = 'negative')
plt.legend()


def happy_60_second(frame): #where is frame
    x_happy.append(datetime.now())
    y_happy.append(JSON_frame['emotion']['happy']['sixty_second_avg'])
    line_happy.set_data(x_happy,y_happy) #this line doesn't work without the comma
    figure.gca().relim()
    figure.gca().autoscale_view()
    return line_happy #or the comma here????

def negative_60_second(frame): #where is frame
    x_negative.append(datetime.now())
    y_negative.append(JSON_frame['emotion']['negative']['sixty_second_avg'])
    line_negative.set_data(x_negative,y_negative) #this line doesn't work without the comma
    figure.gca().relim()
    figure.gca().autoscale_view()
    return line_negative #or the comma here????


x_valence, y_valence = x_datetime, data['y_valence']
x_time = data['x_time']

x_line, y_line = [], []

figure = plt.figure(1)
plt.title('Live Psycological Polarity (Valence) of Reddit Comments')
plt.legend()
line_valence, = plt.plot_date(x_valence, y_valence, '-', color = 'green', label = 'Valence')

def valence_plot(frame):
    x_valence.append(datetime.now())
    x_time.append(time.time())
    valence = float(JSON_frame['emotion']['happy']['sixty_second_avg'] - JSON_frame['emotion']['negative']['sixty_second_avg'])
    y_valence.append(valence)
    saveValence(str(valence)+'\n')
    line_valence.set_data(x_valence, y_valence)

    saveData(x_time, x_valence, y_valence)

    figure.gca().relim()
    figure.gca().autoscale_view()
    return line_valence

#Linear Regression
x_linear, y_linear = [], []
figure = plt.figure(1)
linear_line, = plt.plot_date(x_linear, y_linear, '-', color = 'red', label = 'Linear Regression')
plt.legend()
def linearFit(x_data, y_data, x_test):

    x_data = np.reshape(x_data, [-1,1])
    y_data = np.reshape(y_data, [-1,1])
    x_test = np.reshape(x_test, [-1,1])

    model = LinearRegression()
    model.fit(x_data, y_data)

    y_predicted = model.predict(x_test)

    return y_predicted


def lgr_plot(frame):

    y_predicted = linearFit(x_time, y_valence,time.time())
    #print(y_predicted[0])
    x_linear.append(datetime.now())
    y_linear.append(y_predicted)
    linear_line.set_data(x_linear, y_linear)

    return linear_line



#LSTM

myModel = LSTM_Network('myModel.h5')
try:
    myModel.load_model()
    print('MODEL LOADED')
except Exception as e:
    print(e)
    myModel.define_model()


x_LSTM, y_LSTM = [], []
figure = plt.figure(1)

line_LSTM, = plt.plot_date(x_LSTM, y_LSTM, '-', color = 'purple', label = 'LSTM Prediction', alpha = 0.5)
plt.legend()
def LSTM_Plot(frame):
    try:
        future = (datetime.now() + timedelta(seconds=10)).timestamp()
        prediction = myModel.predict(datetime.now().timestamp())[-1]
        print(prediction)
        x_LSTM.append(datetime.now())
        y_LSTM.append(prediction)
        line_LSTM.set_data(x_LSTM, y_LSTM)

        #Train after predict data point
        myModel.train(data['x_time'][-1],data['y_valence'][-1]) # Only train on single new data piece (Old data is already in model, don't need to retrain), online learning

    except Exception as e:
        print(e)
        print('LSTM plot error')
    return line_LSTM


#animation_happy = FuncAnimation(figure, happy_60_second, interval = 100)
#animation_negative = FuncAnimation(figure, negative_60_second, interval = 100)


animation_valence = FuncAnimation(figure, valence_plot, interval = 10)
animation_zero = FuncAnimation(figure, zero_line_animate, interval = 10)
#animation_lgr = FuncAnimation(figure, lgr_plot, interval = 100)
#animation_LSTM = FuncAnimation(figure, LSTM_Plot, interval = 100)

if __name__ == '__main__':

    Legato_Happy.readScore("i1 1 z") #z innitiates for a long time! Starts running csound thread continuously
    Legato_Negative.readScore("i1 1 z") #z innitiates for a long time! Starts running csound thread continuously

    AskRedditThread.start()
    MetaThread.start()
    ValenceThread.start()
    RollingWindowThread.start()
    LegatoFrameThread.start()

    cPushHappyThread.start()
    cPushNegativeThread.start()

    Legato_Happy.start()
    Legato_Negative.start()
    Happy_LegatoCThread.play()
    Negative_LegatoCThread.play()
    stats_Thread.start()

    stopFlag.set()





    print('Thread Flags Set')
    try:
        while True:
            print('####################### MAIN ########################\n''Program Running @ : ' + time.ctime())
            plt.show()

            time.sleep(5) #main loop sleeping
            #print(JSON_frame['raw']['body'])
            #print(JSON_frame['emotion']['happy']['sixty_second_avg'])
            pass
    except KeyboardInterrupt or Exception as e:
        logging.debug("Broke Main Loop: " + str(e))

        #print(json.dumps(JSON_frame, indent = 4, sort_keys = True))


        LegatoC.stop()
        LegatoC.reset()

        logging.debug('Attempting To clear flags')
        print("Clearing Stopflag")
        stopFlag.clear() #set flag to other threads to break loop
        print("Stopflag Cleared")
        print('Stopping Program Execution')
        exit()
