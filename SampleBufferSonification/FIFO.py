# -*- coding: utf-8 -*- #?????
'Copyright (c) 2019 Jacob Colin Stucki III, All Rights Reserved.'
import os
import os.path #File containgurency cheks
import stat
import subprocess #For FIFO/File DNE Error handling
import sys
import logging #debugggg

class FIFO(object):
    def __init__(self,filePath): #runs at object innitialization
        self.filePath = filePath
        self.exists = os.path.exists(filePath) #I'm still kinda lost in how the constructor is calling it's classes' method on runtime. Does a method exist outside of the instantiated object?
        # I think so... becasue we're calling the method within the class's constructor. LOOK AT doesFIFOExist note
        self.oldline = str()

        logging.debug(str(self.filePath)+" Object Made")
        try:
            if not self.exists:
                self.createFIFO()
            else:
                pass
        except Exception as e:
            logging.debug("Unable to create FIFO")

    def doesFIFOExist(self,filePath): #So Why do we need to call self here? Aren't we just using the method? I guess because it's a reference to an attribute of the class that is called on itself?
        bool = os.path.exists(filePath) #.exists does directory or file, .isfile does file alone
        return bool

    def createFIFO(self): #This must be called within the file doing the reading
        try:
            os.mkfifo(self.filePath) #It exists in working memory, can see with ls -l as a pipe file type, won't show up in finder :)
            print('Made Fifo '+ str(self.filePath))
        except Exception as e:
            print(e)
        pass

    def writeFIFO(self, data): #Can we start to write to the fifo before its created on the inbound file? Need to test
        try:
            fd = os.open(self.filePath, os.O_RDWR) #non-blocking (https://stackoverflow.com/questions/43834991/opening-a-pipe-for-writing-in-python-is-hanging)
            f = os.fdopen(fd, 'w') #also non-blocking
            f.write(data)
            f.write('\n') #PREVENTS MULTI-LINE DATA PIECE. BASICALLY WRITES IN AN EOF
            logging.debug("FIFO: "+str(self.filePath)+ "Wrote: " + str(len(data)))
        except Exception as e:
            logging.debug(str(e))
        pass

    def readFIFO(self):
        fifo = open(self.filePath,'r',encoding='utf8')
        line = fifo.readline()
        fifo.close()
        return line

    def readFIFO_checked(self): # So we can read the fifo line without calling the DataStram obj as a str, just gives options
        checkedLine = str()
        try:
            fifo = open(self.filePath,'r', encoding='utf8')
            line = fifo.readline()
            fifo.close()
            if (str(line) != self.oldline) and (len(line) != 0):
                checkedLine = line
                self.oldline = line
            else:
                checkedLine = str()
        except Exception as e:
            print(e)
        return checkedLine

    def drainFIFO(self):
        fifo = open(self.filePath, 'r', encoding = 'utf8')
        for data in fifo: #Reads all inbound data as fast as it can. Reading the fifo line deletes datapoint, therefore draining the buffered data
            line = fifo.readline()
            if len(line) == 0: #When it returns an EOF, we can say that the FIFO is drained completely.
                break
        fifo.close()
        return print("Pipe Drained")

    def removeFIFO(self):
        os.remove(self.filePath)
        print("Removed Fifo: " + str(self.filePath))
        pass
