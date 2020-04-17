# -*- coding: utf-8 -*- #?????
'Copyright (c) 2019 Jacob Colin Stucki III, All Rights Reserved.'
import time
import ctcsound #kernel
from ctcsound import * #The kernel
from random import randint # :)
import json

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

class Neuron(object):
    """
    Is the data transformation function that lives within the thread that does the I/O of FIFO and data transform
    We pass the static data transformation object becuase fuck super() this is much easier and allows for modularability
        Doing it this way allows us to pass DIFFERENT methods into each CURRENTLY RUNNING object instatiation for every instatiation.
    """
    def __init__(self, name = None, flag = None, transformObj = None, referenceData = None,
                fifo_input = None, fifo_thru = None,file_output = None, fifo_out1 = None,
                fifo_out2 = None, dto_Method = None, frame_obj = None, frame_file = None,
                frame_activity_time = None):
        ## Input Layer
        self.name = name
        self.flag = flag
        self.transformObj = transformObj
        self.referenceData = referenceData #This is data that sits in thread memory, for transform to use (IF NEEDED) This can be any type (list of lists, list, float, int, etc.)
        self.dto_Method = dto_Method #Data transformation object method passed. Allows modularity without having to create a new class for each transform, we just use this class and pass the method

        self.fifo_input = fifo_input

        self.file_output = file_output
        self.fifo_thru = fifo_thru
        self.fifo_out1 = fifo_out1
        self.fifo_out2 = fifo_out2

        self.frame_obj = frame_obj #This is a json object
        self.frame_file = frame_file #This is the last_frame.json FILE (not object) that we write to so we can start up again where we left off when we run the main loop again
        self.frame_activity_time = frame_activity_time #int/float, this is for the activity period carrier update (how many samples were in the buffer, allows for dynamic sampling)
        self.frame_activity_count = 0 # This will be the difference between the two samples. Always starts at 0 and immediately goes to sleep

    def io_FIFO(self):
        logging.debug(str(self.name)+ ' waiting for event flag')
        self.flag.wait() #Wait until event is set to True to start from threading
        logging.debug(str(self.name)+ ' Recieved Go Flag')
        logging.debug(time.ctime() + ' ' + str(self.name)+ " Fifo Transform Started")
        try:
            oldline = str()
            while True:

                if self.flag.is_set() == False: #Check if flag is set to False. If so, break from the loop
                    logging.debug(time.ctime() + ' ' + str(self.name) + ' recieved stop flag')
                    print(time.ctime() + ' ' + str(self.name) + ' BROKE, attempting file append and last queue push')
                    self.file_output.appendFile(output)
                    break

                line = self.fifo_input.get() #updated using Queue
                if (str(line) != oldline) & (len(line) != 0) & (str(line) != '\n'): #This is where the EOF & double read prevent is located, not within the FIFOread method, maybe I'm wrong
                    oldline = line

                    logging.debug(self.name + ' Loading JSON')
                    jObj = json.loads(line)
                    logging.debug(self.name + ' Loaded JSON')

                    logging.debug(self.name + " Attempting FIFO type Transformation")
                    sendToTransform = []
                    sendToTransform.append(jObj)
                    if self.referenceData != None:
                        sendToTransform.append(self.referenceData)

                    transformed_data = self.dto_Method(sendToTransform) #returns a frame
                    logging.debug(self.name + " Completed FIFO type Transformation")

                    output = json.dumps(transformed_data) #dump to string for fifo push, we could possibly make this just a frame object instead of having to dump to string, and basically objectify our IFC (inter-fifo connections) but I want to be lazy and finish the first version first.

                    #Fifo Output

                    logging.debug(self.name + " Attempting FIFO Write")
                    #self.fifo_out1.writeFIFO(output) #old fifo
                    self.fifo_out1.put(output) #new, QUEUE

                    if self.fifo_thru != None: #if we want to pass the data through to another pipe
                        #self.fifo_thru.writeFIFO(line) #OLD
                        self.fifo_thru.put(line) #QUEUE

                    if self.fifo_out2 != None: #If we want to split the data into two FIFOs
                        #self.fifo_out2.writeFifo(output) #Old FIFO
                        self.fifo_out2.put(output) #New QUEUE
                    logging.debug(self.name + " FIFO Write Successful")


                    if self.file_output != None:
                        logging.debug(self.name + " Attempting Log Append")
                        self.file_output.appendFile(output)
                        logging.debug(self.name + " Log Append Success")

        except Exception as e:
            logging.debug("IO Error: "+ str(e))
            print(self.name + " " +str(e) + " Line " + str(sys.exc_info()[-1].tb_lineno)) #https://stackoverflow.com/questions/14519177/python-exception-handling-line-number/20264059

    def io_FRAME(self):
        logging.debug(str(self.name)+ ' waiting for event flag')
        self.flag.wait() #Wait until event is set to True to start from threading
        logging.debug(str(self.name)+ ' Recieved Go Flag')
        logging.debug(time.ctime() + ' ' + str(self.name)+ " FRAME Transform Started")
        try:
            while True:

                if self.flag.is_set() == False: #Check if flag is set to False. If so, break from the loop
                    logging.debug(time.ctime() + ' ' + str(self.name) + ' recieved stop flag')
                    print('stop flag recieved Writing frame file')
                    self.frame_file.writeFile(json.dumps(self.frame_obj))
                    logging.debug('Last frame written')
                    print('Last Frame Written, Breaking')
                    break
                ### vvvvvvvv MAIN LOOP vvvvvvvv
                if self.frame_obj != None: #Check to see if we're passing the frame into the neuron, if we pass the frame it is considered a referential-transform, not JUST a transform. We also need to pass the frame file so that changes ARE ALLOWED TO PERSIST on frame change (may not, may, depends)
                    logging.debug(self.name + " Attempting Frame Write")
                    if self.referenceData != None:
                        tuple = (self.frame_obj, self.referenceData) #has to be an immuteable data type for a method to be able to look at
                        self.frame_obj = self.dto_Method(tuple) #multiparameter
                    else:
                        self.frame_obj = self.dto_Method(self.frame_obj) #One parameter
                    logging.debug(self.name + " Frame Written")

                    if self.frame_file != None: #has to be a sub-frame_obj check because the above frame_obj write puts the data into the frame, and here we're pulling the data from the frame
                        logging.debug(self.name + " Attempting Frame File Write")

                        self.frame_file.writeFile(json.dumps(self.frame_obj)) #Writes the frame_file with the data we want to SAVE for program restart (i.e. relevant memory)
                        logging.debug(self.name + " Frame File Written")
                ### ^^^^^^^ Main Loop ^^^^^^^^

        except Exception as e:
            logging.debug("IO Error: "+ self.name + " " +str(e) + " Line " + str(sys.exc_info()[-1].tb_lineno))
            print(self.name + " " +str(e) + " Line " + str(sys.exc_info()[-1].tb_lineno))

    def io_FRAME_activity(self):
        logging.debug(str(self.name)+ ' waiting for event flag')
        self.flag.wait() #Wait until event is set to True to start from threading
        logging.debug(str(self.name)+ ' Recieved Go Flag')
        try:
            while True:

                if self.flag.is_set() == False: #Check if flag is set to False. If so, break from the loop
                    logging.debug(time.ctime() + ' ' + str(self.name) + ' recieved stop flag')
                    print('stop flag recieved Writing frame file')
                    self.frame_file.writeFile(json.dumps(self.frame_obj))
                    logging.debug('Last frame written, Breaking')
                    print('Last Frame Written, Breaking')
                    break

                ### vvvvvvvv MAIN LOOP vvvvvvvvv
                """
                    We're making an important assumption here, that the frame_obj EXISTS, because otherwise we wouldn't be using this method
                    We don't want to do a statement check within the code because it makes it slow
                    This is a learning process
                    Making assumptions is BAD practice (morally and in code), but it makes the frame sample WICKED FAST, because theres no logic involved
                    This has to do with inhibited inhibitions (Merleau-Ponty/Freud) and the way that humans stereotype
                        The funny thing is that if you acknowledge that you stereotype, it inhibits the part of you that actually does the stereotyping
                        You can then acknowledge that you stereotype, and see past is as just a part of you (characteristic) and not actually a flaw of being
                """
                #sample the frame count
                #pre_sleep_count = self.frame_obj['raw']['count'] #We're going to make this into a class variable, so we only have to check it after the sleep time
                #print("Activity Thread Sleeping")
                time.sleep(self.frame_obj['mean']['phase']) #sleep the monitor for an "update period"

                #sample the frame count again, after the sleep
                """
                for emotion in frame[activity]
                    get sample count
                    compute difference by: sample count - activity (current sample minus the old activity = activity between frame)
                    write activity to frame object

                """


                for emotion in self.frame_obj['mean']['activity']:
                    self.frame_obj['mean']['activity'][emotion]['event_activity'] = self.frame_obj['mean']['activity'][emotion]['event_count'] - self.frame_obj['mean']['activity'][emotion]['event_count_old']
                    self.frame_obj['mean']['activity'][emotion]['event_count_old'] = self.frame_obj['mean']['activity'][emotion]['event_count']

                #sample_count = self.frame_obj['raw']['count']

                #Next, we calculate the difference between the two sample counts and write it to the frame
                #Write to frame object outside of class
                #self.frame_obj['mean']['sample_difference'] = sample_count - self.frame_activity_count #post - pre because sample count increases and we don't want to have another line of code that does an absolute value conversion, faster
                if self.frame_file != None:
                    self.frame_file.writeFile(json.dumps(self.frame_obj)) #calling the file write method on a json dumped string
                #Write to class variable for next comparisson :)
                #print("Sample Count: " + str(sample_count) + " Activity: " + str(self.frame_obj['mean']['sample_difference']))



                    ### ^^^^^^^^ Main LOOP ^^^^^^^^^
        except Exception as e:
            logging.debug("IO Error: "+ self.name + " " +str(e) + " Line " + str(sys.exc_info()[-1].tb_lineno))
            print(self.name + " " +str(e) + " Line " + str(sys.exc_info()[-1].tb_lineno))
        finally:
            return frame
