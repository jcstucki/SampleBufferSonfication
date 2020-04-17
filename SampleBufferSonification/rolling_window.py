import time
import numpy as np
import json
import threading
import random



frame = {'one_second_avg':np.array([])}

class Transformer(object):
    def __init__(self, frame_obj, dto_method):
        self.frame_obj = frame_obj
        self.dto_method = dto_method

    def io(self):
        self.frame_obj = self.dto_method(self.frame_obj)


def generator(frame):
    while True:
        time.sleep(1)
        #generate number of samples
        randint = random.randint(1,1)
        #generate samples
        rand_float = np.random.random(randint)
        frame['generator'] = rand_float
    return frame

def window(frame):
    """
    every one second:
        get generator avg
        put gen avg in one_second_avg
    every 60 seconds/samples
        get one_second_avg (should be 60 samples)
        put sixty_second_avg i
    """
    while True:
        time.sleep(1)
        #get generator average
        gen_avg = np.average(frame['generator'])

        #put gen average in one_second_avg
        one_second_avg = frame['one_second_avg']

        if len(one_second_avg) < 60:
            one_second_avg = np.append(one_second_avg, gen_avg)
        else:
            #left cycle np array
            #https://stackoverflow.com/questions/42771110/fastest-way-to-left-cycle-a-numpy-array-like-pop-push-for-a-queue
            one_second_avg[:-1] = one_second_avg[1:]
            one_second_avg[-1] = gen_avg

        frame['sixty_second_avg'] = np.average(one_second_avg)


        frame['one_second_avg'] = one_second_avg

    return frame

gen_obj = Transformer(frame_obj = frame, dto_method = generator)
window_obj = Transformer(frame_obj = frame, dto_method = window)

gen_thread = threading.Thread(target = gen_obj.io)
window_thread = threading.Thread(target = window_obj.io)

gen_thread.setDaemon(True)
window_thread.setDaemon(True)


def main():
    gen_thread.start()
    window_thread.start()

if __name__ == '__main__':
    main()
    while True:
        time.sleep(1)
        print(frame)
