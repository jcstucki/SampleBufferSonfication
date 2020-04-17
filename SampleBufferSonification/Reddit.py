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

class Reddit(object):
    """Auth is contained within object innitialization"""
    def __init__(self, name, flag, transform, file, fifo, subreddit, frame_obj):

        self.name = name
        self.r = praw.Reddit(user_agent='', client_id='', client_secret="") #calling object creation for Auth
        self.subreddit = subreddit
        self.flag = flag #For thread start/stop PASSING OBJECT
        self.transform = transform
        self.file = file #PASS OBJECT
        self.fifo = fifo #PASS OBJECT


        self.frame_obj = frame_obj #dictionary object #Each one of the "perceptual"/generator/source threads has one of these definitions, the name isn't really relevant, it's just where we can place our sub-level dictionaries


    def pullComment(self):
        try:
            logging.debug(str(self.subreddit)+ ' waiting for event flag')
            self.flag.wait() #Wait until event is set to True to start from threading
            #self.whole['count'] = 0 #reset the count for each perceptual generator on loop begin, each "new frame", the frame_clear is only for when we want to completely reset the frame
            logging.debug(str(self.subreddit)+ ' Recieved Go Flag')
            for comment in self.r.subreddit(self.subreddit).stream.comments(skip_existing=True):
                if self.flag.is_set() == False: #Check if flag is set to False. If so, break from the comment pull loop
                    print(str(self.subreddit) + " recieved STOP flag.")
                    logging.debug('Reddit Stream: '+str(self.subreddit)+ " recieved Stop Flag")
                    break
                    # So here's the deal. It only will break the thread on a NEW comment. We want it to break as soon as the thread flag is changed (Unsure how to)
                    # I think this is the only reasonable way to do it.
                    # If we change the flag back to TRUE when it's in a non-event period, the loop will continue without breaking.
                    # I think this is OKAY.

                ### RAW DATA PULL ###
                rawData=[]

                rawData.append(('count',self.frame_obj['raw']['count'] + 1))
                #rawData.append(('sum', self.frame_obj['raw']['sum'] + self.frame_obj['raw']['count'] ))
                rawData.append(('uuid',str(uuid.uuid4()))) #Unique identifier for each comment (Local based, still random)
                rawData.append(('systime',str(time.time())))
                rawData.append(('sstime',comment.created_utc)) #Time that comment was created on reddit
                rawData.append(('subsphere',self.transform.stripPunctuation(comment.subreddit.display_name)))
                rawData.append(('id',self.transform.stripPunctuation(comment.id))) #ID that reddit gave to comment
                rawData.append(('author',self.transform.stripPunctuation(comment.author.name)))
                rawData.append(('body',self.transform.stripPunctuation(comment.body))) #I hope this workds that'd be cool.
                #We can call a object's method via a passed object!!! That's fuckin DOPE.
                rawData.append(('phase_angle',float(time.time())-float(comment.created_utc)))
                #print('phase_angle: ' + str(float(time.time())-float(comment.created_utc)))
                #Since we're calling the method of a class on a object passed to a class's method, we need to designate the method of the passed class's object call(so the transform Object as well)
                #We changed this so that it no longer needs to be called as method(self, data), because we want to just pass the object's method

                '''
                    So For Some WEIRDDDDD Reason, removing unicode escape individually doesn't work, we have to have a set of permitted characters, and tell it to toss everything else
                    Otherwise we get unicode characters passed through which is no good for our FIFO/Files.
                    We should deal with this later so that we can pass emojis and unicode in order to do stuff to them.
                '''
                raw_dictionary = dict(rawData)#transforms a list of tuples to a dictionary of key:value pairs, then dumps to str

                #print(raw_dictionary['body'])

                #frame['count'] = int(frame['count']) + 1 # for each comment increase the counter
                """it is imperative that we increase the count as we get the data, because out count is used later down the line and I just don't want to change it then because it's easy to change it AS WE GET THE DATA which makes sense because we do it just like our brain does it."""

                self.frame_obj['raw'] = raw_dictionary #put raw dictionary within frame

                 #assumes int, increments by 1 for each comment recieved
                #recalculate sum at the perceptual level, can't calculate it at the end of the transforms, or else we're not reading the sum we're just dumping to the object, since we can only dump the object into the transform neuron once, we need to assign the sum here AS WE GET THE DATA
                    #Using frames makes me wonder if I even need to use FIFO queues..... Because we are referencing an object here that has a structure that we create through past processing steps. and the objects exist outside of the reference frame of the processing step....

                #string = ",".join([str(i) for i in rawData]) #We can change this to transform.encodeSplitSeparator()


        except Exception as e:
            print(e)
        print(str(self.subreddit)+' Broke from thread OK!')
        logging.debug(str(self.subreddit)+' Broke from thread OK!')
