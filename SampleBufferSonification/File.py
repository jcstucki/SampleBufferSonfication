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


class File(object):
    """docstring for FileClass."""
    def __init__(self, filePath):
        #super(FileClass, self).__init__() #This is for child class inheritance. Allows for __init__ reference based upon Parent Class????
        self.filePath = filePath
        self.exists = self.doesFileExist(filePath)

        try:
            if not self.exists: #If file doesn't exist, create it
                self.writeFile('') #Creates file on init
            else: #If file does exist, we just want to load it into the object because we're probably going to read its data or append to it or something and don't want to overwrite it
                pass
        except Exception as e:
            logging.debug(time.ctime()+' File Object Creation Error' + str(e))

    def doesFileExist(self, filePath):
        bool = os.path.exists(filePath) #.exists does directory or file, .isfile does file alone
                # THIS CANNOT BE self.filePath BECAUSE WE'RE CURRENTLY CHECKING IF IT EXISTS TO ASSIGN THE OBJECT TO, OTHERWISE IT WOULD ALREADY BE ASSIGNED
                # We're passing an outside string into the function on innitialization,
                # This function is only neccessary on innitalization because if file deleted during run, would throw exception
        return bool

    def writeFile(self, data): #WRITES WHOLE FILE NO APPEND
        try:
            f = open(self.filePath,'w')
            f.write(data)
            logging.debug(time.ctime() + ' File Written: ' + str(self.filePath))
            f.close()
        except Exception as e:

            logging.debug(time.ctime() + 'ERROR IN FILE WRITE:' + str(self.filePath)+ " " +str(e))
        pass

    def appendFile(self, data):
        f = open(self.filePath, 'a')
        f.write(data)
        f.write('\n') #appends a newline ch to line so there is a new line, have to get rid of this when reading the data
        logging.debug('Appended: ' +str(self.filePath))
        f.close()
        pass

    def readFile(self):
        f = open(self.filePath, 'r')
        allLines = f.read()
        logging.debug('Read File: ' + str(self.filePath))
        f.close()
        return allLines

    def fileToList(self):
        file = open(self.filePath,'r',encoding='utf8') #need encoding for some files. So sure.
        rawData = file.readlines() #Each Line To A List
        noNewLineData=[]
        for line in rawData: #iterates over item in the list
            noNewLineData.append(line[0:-1]) #appends the line (MINUS THE /n NEWLINE CHARACTER) to the new readDatalist
        file.close()
        return noNewLineData
        pass

    def copyFile(self, filePath, copyPath):
        readDataRaw = self.fileToList(filePath)
        f = open(filePath,'r')
        fp.writelines(readDataRaw)
        fp.close()
        pass

    def clearFile(self, filePath):
        try:
            open(fileName,'w').close() #opening in write mode, writing no data to it, then closing, clears the file
        except:
            e = Exception
            print('Failed to clear File with Error: %s', e)
        pass
