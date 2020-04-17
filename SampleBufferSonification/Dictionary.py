# -*- coding: utf-8 -*- #?????
'Copyright (c) 2019 Jacob Colin Stucki III, All Rights Reserved.'

class Dictionary(object):
    """yes"""
    def __init__(self, name = None, type = None, fileList = None):
        self.name = name
        self.type = type #JSON, \n file, text file, etc. MySQL
        self.fileList = fileList

    def getName(self):
        gotName = self.name
        return gotName

    def getList(self):
        gotList = self.fileList
        return gotList
