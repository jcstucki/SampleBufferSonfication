import json
from datetime import datetime
import time


def saveData(x_time, x_valence, y_valence):
    x_valence_str = []
    for i in range(0,len(x_valence)):
            str = x_valence[i].strftime('%Y-%m-%d %H:%M:%S.%f')
            x_valence_str.append(str)
    dict = {'x_time':x_time, 'x_valence':x_valence_str, 'y_valence':y_valence}
    with open("data.json", 'w') as write_file:
        json.dump(dict, write_file)

def loadData():
    try:
        with open("data.json", "r") as read_file:
            data = json.load(read_file)
    except:
        data = {'x_time':[],'x_valence':[],'y_valence':[]}
    return data



def saveValence(string):
    with open("valence.json", "w") as write_file:
        write_file.write(string)
    pass
