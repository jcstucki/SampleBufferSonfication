# -*- coding: utf-8 -*- #?????
'Copyright (c) 2019 Jacob Colin Stucki III, All Rights Reserved.'
import time
import ctcsound
from ctcsound import *
import sys
import logging #debugggg

class cPush(object):
    """
        This is the object that sits in the thread, much like the neuron
        It does the communication between the frame processing threads and the running csound threads
    """
    def __init__(self, name = None, legatoObj = None, frame = None, emotion = None, flag = None):
        self.name = name
        self.legatoObj = legatoObj #CSOUND OBJECT
        self.flag = flag
        self.frame = frame
        self.emotion = emotion

    def legatoPush(self):
        try:
            logging.debug(str(self.name)+ ' waiting for event flag')
            self.flag.wait() #Wait until event is set to True to start from threading
            logging.debug(str(self.name)+ ' Recieved Go Flag')
            while True:

                if self.flag.is_set() == False: #Check if flag is set to False. If so, break from the loop
                    logging.debug(time.ctime() + ' ' + str(self.name) + ' recieved stop flag & broke')
                    break

                emotion_dict = self.frame['legato'][self.emotion]
                #print(emotion_dict)
                self.legatoObj.setControlChannel("kAmp", emotion_dict['amp'])
                self.legatoObj.setControlChannel("kFundamentalFreq", emotion_dict['fund_freq'])
                self.legatoObj.setControlChannel("kFundamentalMod", emotion_dict['fund_mod'])
                self.legatoObj.setControlChannel("kModFreq", emotion_dict['mod_freq'])
                self.legatoObj.setControlChannel("kModDepth", emotion_dict['mod_depth'])
        except Exception as e:
            logging.debug("cPush: "+ self.name + " " +str(e) + " Line " + str(sys.exc_info()[-1].tb_lineno))
            print(self.name + " " +str(e) + " Line " + str(sys.exc_info()[-1].tb_lineno))
        pass
