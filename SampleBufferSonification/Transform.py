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


class Transformer(object):
    def __init__(self, frame_obj, dto_method):
        self.frame_obj = frame_obj
        self.dto_method = dto_method

    def io(self):
        self.frame_obj = self.dto_method(self.frame_obj)






class DataTransform(object):
    """

    So none of the methods contained within this class are called on the object created by the class, they are called
        as functions upon data. We only need to innitiate one object of this class's type, because the object will be passed to different
        other object methods.

    i.e. It transforms OTHER object's data, not its own data, so calling self for a method is irrelevant and just more confusing code in the passed object method call
            The Data Transform realized object is just a container that contains all of the data transformation. Not an object that has a pointer.

        The above is a lie, this is a meta-stagnant class to which we will be riding a "Reader" sub-class. We will have a method is called on object "run"/"status/etc"
        which we can overwrite with the dataTransformation method we want to use.

    """
    def showMe(tuple):
        print(tuple[0])
        print(tuple[1])
        pass

    def count(list):
        counter = 0
        output = []
        for item in list:
                counter += 1
        output.append(counter)
        return output

    def count2(list):
        output = len(list)
        return output

    def availAtrib(self): #Makes it easy to see what data we have!!!!
        vars = vars(self)
        return vars

    def encodeSplitSeparator(list, separator):
        string = separator.join([str(i) for i in list])
        return string

    def separatorToList(line,separator): #separator is a string, we use ","
        data = []
        data = line.split(separator) # pull the string from the line being read and split it into a new list
        return data

    def SI(input,reference):
        common = set(input).intersection(reference)
        return common

    def removeSeparatorFromRaw(input):
        separator = ","
        output = "".join(c for c in input if c not in separator)
        return output

    def removeUnicode(input):#Writing this because it's dumb and I just want to stop the issue. FML. THIS NEEDS TO BE DEALT WITH IF I WANT TO DO EMOJI RECOGNITION
        unicode_char = "\\" #Double because otherwise we get a " escape.
        #output = "".join(c for c in input if c not in unicode_char)
        output = "".join(i.replace('\\','') for i in input)
        return output

    def stripPunctuation(input):
        permitted = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\ '" #leaves in appostrophie, we're using commas to separate data
        output = "".join(c for c in input if c in permitted) #List comprehension [transform for item in list if conditional]
        #Need to make it so that this replaces all punctuation with a ' ' so that it's not combining words together in order to process more words
        #punctuation modifier
        return output

    def tokenizeString(string):
        tokens = word_tokenize(string)
        numOfTokens = len(tokens)
        return numOfTokens, tokens #returns number of tokens and the tokens themselves

    def compareLists(input,reference,returnwords):
        commonWords = set(input).intersection(reference)
        if (returnwords == True) & (len(commonWords) != 0):
            return len(commonWords), commonWords #returns the number of matches as well as the matches
        else:
            return len(commonWords)

    def dictListCompare(recieve): #recieve is always a list
        #object is in first index
        #reference data is in second index
        frame = recieve[0]
        listOfDictObjs = recieve[1]

        body = frame['raw']['body'] #grabs value from key in dictionary

        count = frame['count']

        tokens = word_tokenize(body)
        for dictionary in listOfDictObjs:
            dictionaryList = dictionary.getList() #calls instantiated Dictionary Object method .list()
            commonWords = set(tokens).intersection(dictionaryList) # a list of the common words
            numberCommon = len(commonWords)
            count[dictionary.getName()]= numberCommon
            #I just have to trust that my dictionaries are correct at this point, counts seem weird but that can be handled with a single JSON dictionary which I don't want to deal with right now

        frame['count'] = count #dump relevant dict obj to jObj

        return frame

    def normalizeEmotion(frame): #got the normalize term from the python Markofv Chain program. I think it just means going from counts to percentage, which is what we're doing here.

        count = frame['count'] #pull dictionary object from full jboj dictinary

        elementTotal = count['real'] #because it encodes to a string for inter-fifo (synapse), we provide the interface with int()
        happyElementTotal = count['happy']
        negativeElementTotal = count['negative']

        #valence = dict() #create dictionary object WE DONT NEED TO DO THIS
        valence = frame['valence']

        emotionalElementTotal = happyElementTotal + negativeElementTotal #always add emotion in pairs (plutchiks)

        try: #Word total might be zero (unlikely but can happen and don't want it to cause an exception)
        	percentEmotion = emotionalElementTotal / elementTotal# Total emotional Elements over element Total = percentage of emotion per element
        except:
        	percentEmotion = 0

        try:#might be a  DIVIDE BY ZERO ERROR because there might be zero emotional elements
        	happyPercentage = happyElementTotal / elementTotal
        except: #if there are zero, we say the percentage is zero
        	happyPercentage = 0

        try: #might be a  DIVIDE BY ZERO ERROR because there might be zero emotional words in the comment
        	negativePercentage = negativeElementTotal / elementTotal
        except:
        	negativePercentage = 0 #If no emotional words, set zero

        valence['total'] = percentEmotion
        valence['happy'] = happyPercentage
        valence['negative'] = negativePercentage

        frame['valence'] = valence #put valence obj back into full jObj dictionary

        #For sb-son

        #get gen arrays
        happy_generator = frame['emotion']['happy']['generator']
        negative_generator = frame['emotion']['negative']['generator']

        #append valence
        if frame['raw']['uuid'] != frame['uuid_old']:
            happy_generator = np.append(happy_generator, np.array(happyPercentage))
            negative_generator = np.append(negative_generator, np.array(negativePercentage))
            frame['uuid_old'] = frame['raw']['uuid']

        #Place back in frame
        frame['emotion']['happy']['generator'] = happy_generator
        frame['emotion']['negative']['generator'] = negative_generator



        return frame #return full jObj dictionary

    def eventCounter(frame):

        valence = frame['valence']
        noTotal = dict()
        for key, value in valence.items():
            if key != 'total':
                noTotal[key] = value #we now have a dictionary of emotion:percent ONLY, without totalEmotionalPercentage within the dictionary, which we don't need for this function since we are counting events of "type" left or right

        # https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
        max_key = max(noTotal, key = lambda key: noTotal[key])
            #This uses a lambda function, which is a non-defined function. small function to return something. In our case it returns the value of the key in the dictionary
            #Then, the max function takes two args, the first is the value to be returned, the second is the
            #"value that should be used for ranking" aka we are using the value as the key/input
            #Since we only want the name of the key, and not the value, the lambda key:noTotal[key] returns the value, and uses the value to rank noTotal keys
        #print(max_key)

        # KEEP IN MIND, WE ARE THROWING AWAY VALUES THAT NULLIFY EACH OTHER HERE
        # SO WHEN THERE IS A CANCELATION, WE ARE NOT INCREMENTING ANYTHING, BUT THAT DATA POINT PROBABLY HAS USEFULNESS I'M JUST BEING FUCKING
        # DIRTY AS SHIT RIGHT HERE IN ORDER TO GET TO AN OUTCOME FASTER


        frame['mean']['activity'][max_key]['event_count'] += 1

        """
        for emotion in valence (THATS NOT THE TOTAL PERCENTAGE, we want to ignore that one)
            Quick and dirty to remove the total emotion we can create a dictionary here that copies the valence dictinary without the total percentage
                Otherwise we have to do some weird heirarchical structuring view which makes the code more complex and right now we're just going for
                OUTCOME
            find the emotion with the maximum percentage
            increment the emotion_count KEY:value by 1

        OUTSIDE OF method:

        """

        return frame


    def decorrelateList(input): #This function returns a tuple with the data and original index
        inputCopy = input #We need this so we can remove items from the copied set (instead of the actual input set) so we know how many items we have left to decorrelate. Is the "To-Do" list
        decorrelatedIndexDataTupleList = []
        indexDataTupleList = [] #format of (index,data) so that it's easier to read when we append some auto/cross correlated data

        for index in range(0,len(inputCopy)): #This loop appends the original index of the data, dataIndexTupleList is still correlated, just added another data point to it :)
            tupple = (index,inputCopy[index])
            indexDataTupleList.append(tupple)

        print("Assigned Index Tuple:\n"+str(indexDataTupleList))

        while len(indexDataTupleList) > 0: #We changed this to a while loop from a for loop, but I think in reality it can actually be both
            randomIndex = randint(0,len(indexDataTupleList)-1) # Picks a "random" index to be set as the next, this needs to be len(inputCopy)-1 because we're looking at index value, and it starts at 0, but length returns all numbers
            #It was looking at 0,1,2,3,4,5,6,7,8,9,10 (11 indecies), while we want 0,1,2,3,4,5,6,7,8,9 (10 indicies)
            decorrelatedIndexDataTupleList.append(indexDataTupleList[randomIndex])
            del indexDataTupleList[randomIndex] #after we use the index, we delete it from the "todo" list
        return decorrelatedIndexDataTupleList #If we want to go back to the original order of list, we also need to output a list that is a 1:1 match of the decorrelated data, but with the original index

    def recorrelateTuple(input): #input must be a list of tuples with the tuple[0] being the original index
        """
            Q: Why do we want to be able to recorrelate our data if we're doing autocorrelation on it to find the fundamental?
                Does autocorrelation return a single value for a SET of data, or does it return a single value for a PIECE of data?
                I have a suspiscion it returns a single value for a piece of data. Because if we're sonifying, we are going to assign that value to
                one of our mapped parameters. (i.e. vibratto)
                It is the correlation of the signal to itself, but with time delay. So for each datapoint/vector, we want to compute the auto-correlation with delay t
                and then ADD that value to the time-correlated signal(the original meta-data) in order to sonify the AC signal over the original sample time.
                SO, because the autocorrelation is used to find the similarity in the decorrelated data, we'll want to put the decorrolated data back in it's original order
                while the autocorrelation is carried with it.
        """
        toDo = input #copy the input so we can remove from it as we do work
        inputLength = len(inputLength) #So we can look at the max data index, and do set().intersection() for the next index to append
        recorrelatedList = []
        for item in toDo:
            pass

        pass

    def wi_Mean(frame): #Takes a list of objects
        """
            https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#On-line_algorithm
            # SOURCE:
            #   https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#On-line_algorithm
            #I don't fully comprehend this.
            #I understand some of the  outcome, and SORT OF the math itself, but not completely.
            #I'm gonna keep looking at it.
            #That's why I'm sourcing HARD.
            # Weightings should be based off of statistical significance of the element.
            # That makes sense right?
            #This is based off of a bunch of research into incremental averaging.
            #   https://math.stackexchange.com/questions/106700/incremental-averageing
            #   https://dsp.stackexchange.com/questions/811/determining-the-mean-and-standard-deviation-in-real-time

        """
        #json dict always first, it is the top tier encompassing structure
        #referenceData = recieve[1] #reference dict always second #Not using reference data, well we are but it's carried within the jObj. I could possibly do it with the valenced dictionary in the future, but who knows????

        mean = frame['mean'] #is a dict object


        #These two are the same for each calculation
        newWeight = frame['raw']['count'] #Current element number (count is constantly increasing)
        #print("count/weight " + str(newWeight))

        oldWeight = frame['raw']['sum'] #Sum of all weights before (HUGGGGGEEEE number)
        #print("sum " + str(oldWeight))

        #We don't need to create the dictionary object here because we're assuming it already exists (existance value of 1)

        for key in frame['mean']['emotion']:
            #Datapoints we need
            oldMean = frame['mean']['emotion'][key] #since we didn't define emotion as a dictionary object, and we're iterating over it, we need to use frame['emotion'][key] to assign the mean to the value within the emotion frame
            newDataPoint = frame['valence'][key] #percent (percentEmotion, happyPercentage, negativePercentage)

            #Datapoints we calculate
            deltaMean = newDataPoint - oldMean #The difference between two scalars (TYPE similarity, they are both measured percentages, both normalized percentage vectors)
            try:
                weightingRatio = newWeight / oldWeight #The new weighting ratio. the ratio of change. the new multiplier(weight) compared to the sum of all multipliers(weights) A normalized number
            except:
                weightingRatio = 1 #if we get a divide by zero error (notably on the first iteration), weighting is 1.

            newMean = oldMean + weightingRatio * deltaMean #newMean = oldMean + ( newWeight / oldWeight ) * (newDataPoint - oldMean)

            frame['mean']['emotion'][key] = newMean #since we didn't define emotion as a dictionary object, and we're iterating over it, we need to use frame['emotion'][key] to assign the mean to the value within the emotion frame


        #Interim Processing
        #newWeightedSum = oldWeight + newWeight #SO WE CANT RECALCULATE THE SUM AT THE END OF THE TRANSFORMS, BECAUSE ITS NEEDED CONCURRENTLY AS WE DO THE TRANSFORM, SO BOTH THE COUNT AND SUM CHANGE AT PERCEPTUAL/GENERATOR LEVEL
        #print("new sum " + str(newWeightedSum))

        #Return These Values
        #mean['sum'] = newWeightedSum
        #print(json.dumps(mean, indent = 4, sort_keys = True))

        frame['mean'] = mean #dump frame dict obj back into total jObj

        '''
        #Dunno if we need these
        oldWeightSquare = 0
        S = 0

        #newWeightedSumSquare = (oldWeight * oldWeight) + (newWeight * newWeight) #For Sample Reliability Things

        for x, w in dataWeightPairs:  # Alternatively "for x, w in zip(data, weights):"
            newWeightedSum = oldWeight + w # oldWeight + newWeightedSum
            newWeightedSumSquare = oldWeightSquare + w * w # oldWeightSquare + newWeightedSumSquare, we are squaring because
            newMean = oldMean + (w / oldWeight) * (x - meanOld)
            S = S + w * (x - oldMean) * (x - newMean) #sample variance, aka how much our data point differs from the old average (x - meanOld) * (x - mean)

        population_variance = S / oldWeight #How different the new
        # Bessel's correction for weighted samples
        # Frequency weights
        sample_frequency_variance = S / (oldWeight - 1)
        # Reliability weights
        sample_reliability_variance = S / (oldWeight - oldWeightSquare / oldWeight)
        '''
        return frame

    def mean_phase(frame): #Takes a list of objects

        #json dict always first, it is the top tier encompassing structure
        #referenceData = recieve[1] #reference dict always second #Not using reference data, well we are but it's carried within the jObj. I could possibly do it with the valenced dictionary in the future, but who knows????

        mean = frame['mean'] #is a dict object


        #These two are the same for each calculation
        newWeight = frame['raw']['count'] #Current element number (count is constantly increasing)
        #print("count/weight " + str(newWeight))

        oldWeight = frame['raw']['sum'] #Sum of all weights before (HUGGGGGEEEE number)
        #print("sum " + str(oldWeight))

        #We don't need to create the dictionary object here because we're assuming it already exists (existance value of 1)


        #Datapoints we need
        oldMean = frame['mean']['phase'] #since we didn't define emotion as a dictionary object, and we're iterating over it, we need to use frame['emotion'][key] to assign the mean to the value within the emotion frame
        newDataPoint = frame['raw']['phase_angle'] #percent (percentEmotion, happyPercentage, negativePercentage)

        #Datapoints we calculate
        deltaMean = newDataPoint - oldMean #The difference between two scalars (TYPE similarity, they are both measured percentages, both normalized percentage vectors)
        try:
            weightingRatio = newWeight / oldWeight #The new weighting ratio. the ratio of change. the new multiplier(weight) compared to the sum of all multipliers(weights) A normalized number
        except:
            weightingRatio = 1 #if we get a divide by zero error (notably on the first iteration), weighting is 1.

        newMean = oldMean + weightingRatio * deltaMean #newMean = oldMean + ( newWeight / oldWeight ) * (newDataPoint - oldMean)

        frame['mean']['phase'] = newMean #since we didn't define emotion as a dictionary object, and we're iterating over it, we need to use frame['emotion'][key] to assign the mean to the value within the emotion frame


        #Interim Processing
        #newWeightedSum = oldWeight + newWeight #SO WE CANT RECALCULATE THE SUM AT THE END OF THE TRANSFORMS, BECAUSE ITS NEEDED CONCURRENTLY AS WE DO THE TRANSFORM, SO BOTH THE COUNT AND SUM CHANGE AT PERCEPTUAL/GENERATOR LEVEL
        #print("new sum " + str(newWeightedSum))

        #Return These Values
        #mean['sum'] = newWeightedSum
        #print(json.dumps(mean, indent = 4, sort_keys = True))

        frame['mean'] = mean #dump frame dict obj back into total jObj

        return frame

    def showStats(frame):
        while True:
            if frame['raw']['uuid'] != frame['uuid_old']:
                time.sleep(1)
                print('Comment Count: '+ str(frame['raw']['count']))
                #print(frame['raw']['subsphere'] + '-' + frame['raw']['body'])

    def legato_map(frame):

        # frame['legato'] = dict() #omitting this because we're going to put it in our default frame_restructure
        # WE'RE GONNA BE DIRTY HERE TOO AND REMOVE THE TOTAL DICTIONARY FROM THE EMOTIONS BECAUSE WE DONT REALLY NEED IT RIGHT NOW
        try:
            emotions = frame['mean']['emotion']
            for key, value in emotions.items(): #we do this because there will be individual carriers for each emotion. This is kinda similar to an EKG, but emotional. And subjective for the self.
                #Might be possible to change this so that we look at key:keyname or something so that it's much easier to add emotions
                # So that it's not doing a key, value thing and we just map from wherever the parameter is in the frame
                if emotions[key] != 'total': #Dirty trick
                    #frame['legato'][key] = dict() #omitting this because we're going to put it in our default frame_restructure

                    #clears/defines the dict on every new frame, since I don't think we'll be doing calculations based on the old values
                    #Although that might be a possibility. Unsure right now.
                    #We're not going to do it that way, we want access to the previous values for shits and giggles.

                    emotional_frequencies = {
                        'happy':1000,
                        'negative':400
                    }

                    vibrato_depth ={
                        'happy':1,
                        'negative':2

                    }

                    equal_loudness = {
                        'happy':1,
                        'negative':3
                    }

                    #Ratio Modifiers i.e The weighting of the parameter mappings, these are somewhat subjective values and must be played with (can multisample, emperical testing)
                    ratio_Amp = 2
                    ratio_Fund_Freq = 1
                    ratio_Fund_Mod = 1
                    ratio_Mod_Freq = 1
                    ratio_Mod_Depth = 100

                    """
                    """

                    #Parameter Mapping

                    ### Copyright Jacob C. Stucki III ###

                    parameter_Amp = frame['emotion'][key]['one_second_avg'][-1] * equal_loudness[key] #Intuition tells me this should be mapped to the most dominant emotion in the frame, we basically already do this, as our mean values will always reflect the dominant emotion
                    #parameter_Amp = equal_loudness[key]
                    parameter_Fund_Freq = emotional_frequencies[key]  #Intuition tells me this should be mapped to portrey the emotion itself. Sort of like Major/Minor being happy/sad respectively, intuition tells me this can be done for frequency too!
                    parameter_Fund_Mod = 1 #No, this should not be modified at all, because we don't want to modify the centerband frequency
                    parameter_Mod_Freq = frame['emotion'][key]['sixty_second_avg'] * vibrato_depth[key]# Intuition says activity. This and the below parameters are difficult
                    #parameter_Mod_Freq = emotions[key] #Alternate mapping of modulator freq to amplitude of emotion frequency = amplitude
                    parameter_Mod_Depth = 1 #get last one second avg from one second avg list

                    #Dictionary Assigns
                    frame['legato'][key]['amp'] = ratio_Amp * parameter_Amp
                    frame['legato'][key]['fund_freq'] = ratio_Fund_Freq * parameter_Fund_Freq
                    frame['legato'][key]['fund_mod'] = ratio_Fund_Mod * parameter_Fund_Mod
                    frame['legato'][key]['mod_freq'] = ratio_Mod_Freq *  parameter_Mod_Freq
                    frame['legato'][key]['mod_depth'] = ratio_Mod_Depth *  parameter_Mod_Depth


        except Exception as e:
            logging.debug("Legato Error "+ " " +str(e) + " Line " + str(sys.exc_info()[-1].tb_lineno))


        return frame

    def delta_mean_depth(frame): #outputs an ever increasing small float, and a normalized number that we can use to map to the depth so that it stays relatively the same

        for key, value in frame['mean']['emotion'].items():
            old = frame['mean']['delta'][key]['old']
            new = frame['mean']['emotion'][key]
            delta_depth = abs(old - new) #minus because change, not additive
            frame['mean']['delta'][key]['delta'] = delta_depth #write the frame
            frame['mean']['delta'][key]['old'] = frame['mean']['emotion'][key] #set old mean to current after delta calculation for next value
        return frame

    def depth_to_normal(frame): #I believe we can say that deltaMean is always going to get smaller because since we're creating an average, the change in the average will be smaller due to our
                                    #ratioOfChange calculation always becoming smaller and smaller, each data point has less and less of an effect, so the delta will always get smaller
                                    #we don't want our delta depth to get smaller because we always want to hear the depth as a perceptually "stable" aka the of the same power
                                    #of course the value will change but we don't want to have to change our ratio mapping factor because that determines how pronounced the depth is
                                    #very different. I believe this calculation is required.

        for key, value in frame['mean']['emotion'].items():
            delta_string = str(frame['mean']['delta'][key]['delta'])

            if int(delta_string[0]) == 0: #if the first digit is zero we have a float that's not time some power of ten so we need to take off the leading zeros
                """
                set interim string point,
                remove first two indicies (0.)
                for remaining indicies check if value (not character) is equal to zero, if it is move to next index
                if index value is larger than zero, we have our first digit
                create new string,
                place first digit in zero index, place '.' in second index, and for rest of remaining digits in index, append string
                """
                interim_string = delta_string[2:] #remove 0. [2:] removes the first two indecies
                marker = int() #assign so that we can append to it, refernce before assignment
                for index in range(0,len(interim_string)): #for remaining indcies from left most index to right most index
                    #while int(interim_string[index]) == 0: #while it's still zero continue, WE ALSO HAVE TO INT THE VALUES HERE BECAUSE WE'RE ITERATING OVER A STRING
                    if int(interim_string[index]) != 0: #if it's any value other than zero we have our first digit
                        marker = index
                        break
                decimalized_string = [] #for adding back the . so we have a nice clean depth number :)
                decimalized_string.append(str(interim_string[marker])) #give first real digit of interim_string to decimal list
                decimalized_string.append('.') #second index is the decimal
                for index in range(marker + 1, len(interim_string)): #for each index between the first real number + 1 (because we used that up when we assigned the first index) and the last index in the floating point
                    decimalized_string.append(interim_string[index])
                normal_string = float(''.join(decimalized_string)) #joins the list into a string, and floats the string
            elif int(delta_string[0]) > 0: # if the first index is greater than zero we have a float that has a power of ten, aka an "e" character in it
                for i in range(0,len(delta_string)): #iterate the index
                    if delta_string[i] == "e": #check the index value
                        split = i #index to split value
                normal_string = delta_string[0:split] #return the 0th index to the "e" character

            normal_float = float(normal_string)
            frame['mean']['delta'][key]['delta_normal'] = normal_float

        return frame



    def rolling_window(frame):
        while True:

            for key in frame['emotion']:

                time.sleep(0.5)
                #get generator average
                try: #mean of empty slice error handling
                    gen_avg = np.average(frame['emotion'][key]['generator'])
                    frame['emotion'][key]['generator'] = np.array([0]) #reset the generator array after average (NECESSARY, or else just keeps appending)
                except:
                    pass
                #put gen average in one_second_avg
                one_second_avg = frame['emotion'][key]['one_second_avg']

                if len(one_second_avg) < 60:
                    one_second_avg = np.append(one_second_avg, gen_avg)
                    print('Buffering: '+str(len(one_second_avg))+"/60")
                else:
                    #left cycle np array
                    #https://stackoverflow.com/questions/42771110/fastest-way-to-left-cycle-a-numpy-array-like-pop-push-for-a-queue
                    one_second_avg[:-1] = one_second_avg[1:]
                    one_second_avg[-1] = gen_avg

                try:
                    frame['emotion'][key]['sixty_second_avg'] = np.average(one_second_avg)
                except:
                    pass

                frame['emotion'][key]['one_second_avg'] = one_second_avg

        return frame
