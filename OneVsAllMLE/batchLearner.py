from helper import helper
import matplotlib.pyplot as plt
import copy
from softZeroOneLoss import softzeroone
from MLEOneVsAll import mleonevsall
from hingeLoss import hinge
import numpy as np
import random
import os
import time
from neuralnetwork import neuralnetwork
import sys


class batchLearner():
    CLASSES = ["sitting", "walking", "standing", "standingup", "sittingdown"]
    helper = None

    filenameTemplate = "../output/parameterOptimize/par_"
    def __init__(self):
        self.helper = helper()


    # Range is in [min, max,function, functionparameters)
    def optimizeParameters(self, classifier, parameterStart, optimizeID, parRange, trainingSamples, startWeights, maxSteps, maxNonChangingSteps, folds, parameterName, stepFunction):
        historicalAccuracy = []
        parameterHistory = []
        accMax = 0
        parMax = parRange[0]
        pOptimize = parRange[0]
        timeStart = time.time()

        #the initial step size is set to the minValue
        stepSize = parRange[0]

        while (pOptimize < parRange[1]):
            parameterHistory.append(pOptimize)
            print("Testing parameter value " + str(pOptimize))
            curParameters = copy.deepcopy(parameterStart)
            curParameters[optimizeID] = pOptimize

            curAcc = self.crossValidate(classifier, curParameters, trainingSamples, startWeights, maxSteps, maxNonChangingSteps, folds)
            self.printCurAcc(parameterName, pOptimize, curAcc, classifier.__name__, timeStart)
            historicalAccuracy.append(curAcc)
            if (curAcc > accMax):
                accMax = curAcc
                parMax = pOptimize

            #see if we need to adjust the step size
            pOptimize, stepSize = stepFunction(pOptimize, stepSize)

        print("Optimized parameter " + str(optimizeID) + ". Best result: " + str(curAcc) + " for value " + str(parMax))
        #plt.plot(parameterHistory, historicalAccuracy)
        self.printParameterToFile(parameterName, parMax, parameterHistory, historicalAccuracy, classifier.__name__)
        return parMax, accMax

    def printCurAcc(self, parameterName, pOptimize, curAcc, className, timeStart):
        print("Writing parameters.")
        filename = self.filenameTemplate + className + "_" + parameterName + "_DEBUG_" + str(timeStart) + ".txt"
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        file = open(filename, "a+")
        file.write(str(pOptimize) + "; " + str(curAcc) + "\n")
        file.close()


    def printParameterToFile(self, parameterName, parMax, parHistory, accHistory, className):
        print("Writing parameters.")
        filename = self.filenameTemplate + className + "_" + parameterName + ".txt"
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        file = open(filename, "a+")
        file.write(parameterName + ": " + str(parMax))
        file.write("\n")
        for i in range(len(parHistory)):
            file.write(str(parHistory[i]) + "; " + str(accHistory[i]) + "\n")
        file.close()

    def crossValidate(self, classifier, parameters, trainingSamples, startWeights, maxSteps, maxNonChangingSteps, folds):
        trainingSets = self.splitTrainingSet(trainingSamples, folds)
        listAccs = []
        for i in range(folds):

            classifierInstance = classifier(self.CLASSES, maxSteps, maxNonChangingSteps, parameters)
            classifierInstance.learn(self.getCrossValidationSet(trainingSets, i), startWeights)
            learnedWeights = classifierInstance.maxWeights
            curAcc = 1-self.helper.calcTotalError(classifierInstance, trainingSets[i], learnedWeights)[0]
            listAccs.append(curAcc)
        print listAccs
        avgAcc = sum(listAccs)/folds
        print "Average accuracy: " + str(avgAcc)
        return avgAcc

    def getCrossValidationSet(self, crossValidationSets, k):
        returnSet = []
        for i in range(0, len(crossValidationSets)):
            if (i != k):
                returnSet += crossValidationSets[i]
        return returnSet

    def splitTrainingSet(self, trainingSamples, crossvalidationNumber):
        returnList = []
        for i in range(crossvalidationNumber):
            returnList.append([])
        for sample in trainingSamples:
            returnList[random.randint(0, crossvalidationNumber-1)].append(sample)
        return returnList


CLASSES = ["sitting", "walking", "standing", "standingup", "sittingdown"]

def plus(p, list):
    return p + list[0], list

def pot(p, list):
    return p * list[0]. list

def optLogarithmic(p, stepSize):
    pNew = p + 2*stepSize
    if (pNew >= 10*stepSize):
        stepSize *= 10
        pNew = stepSize
    return pNew, stepSize

def optLogarithmicInverse(p, list):
    return p

stepFunction = optLogarithmic
#
# batch = batchLearner()
# helper = helper()
# classifier = hinge
# parameterStart = [1e-5, 0.98, 2, 0]
optimizeID = 1
parRange = [1e-5, 1e-1, stepFunction, [1e-5]]
# trainingSample = helper.readData("../data/dataset-complete_90PercentTrainingSet_mini10Percent_standardized.arff")
# startWeights = []
# for i in range(len(CLASSES)):
#     dummyWeight = np.zeros(17)
#     startWeights.append(dummyWeight)
#
# maxSteps = 70
# maxNonChangingSteps = 8
#

CLASSES = ["sitting", "walking", "standing", "standingup", "sittingdown"]
CLASSIFIERS = {'MLE': mleonevsall, 'SOFTZEROONE': softzeroone, 'HINGE': hinge, 'NN': neuralnetwork}
STEPFUNCTIONS = {'logstep': optLogarithmic, 'logstepInverse': optLogarithmicInverse}

MAXSTEPS = 70
MAXNONCHANGINGSTEPS = 10
PARAMETERS = None
parName = ""
folds = 10

classifier = None

#create start weights or read them
startWeights = None
trainingSample = None
helperInstance = helper()
filename = "../data/dataset-complete_90PercentTrainingSet_mini10Percent_standardized.arff"

#read cmd line arguments
print("About to checking arguments...")
for i in range(len(sys.argv)):
    if sys.argv[i] == "-h" or sys.argv[i] == "--help":
        print("Here should be your help.")
        sys.exit()
    elif sys.argv[i] == "-c" or sys.argv[i] == "--classifier":
        classifier = CLASSIFIERS[sys.argv[i+1]]

    elif sys.argv[i] == "-p" or sys.argv[i] == "--parameters":
        print("received parameters - processing them...")
        stringList = sys.argv[i+1]
        PARAMETERS = [float(x) for x in stringList.split(',')]
        print PARAMETERS

    elif sys.argv[i] == "-ms" or sys.argv[i] == "--maxSteps":
        MAXSTEPS = int(sys.argv[i+1])
        print("Max Steps: " + str(MAXSTEPS))


    elif sys.argv[i] == "-mncs" or sys.argv[i] == "--maxNonChangingSteps":
        MAXNONCHANGINGSTEPS = int(sys.argv[i+1])
        print("Max non changing Steps: " + str(MAXNONCHANGINGSTEPS))

    elif sys.argv[i] == "-oID" or sys.argv[i] == "--optimizeID":
        optimizeID = int(sys.argv[i+1])
        parName = sys.argv[i+2]
        print("PArameter to optimize: " + str(parName) + " ID: " + str(optimizeID))

    elif sys.argv[i] == "-pS" or sys.argv[i] == "--parameterStart":
        parRange[0] = float(sys.argv[i+1])
        print("start optimize: " + str(parRange[0]))
    elif sys.argv[i] == "-pE" or sys.argv[i] == "--parameterStop":
        parRange[1] = float(sys.argv[i+1])
        print("stop optimize: " + str(parRange[1]))

    elif sys.argv[i] == "-f" or sys.argv[i] == "--folds":
        folds = int(sys.argv[i+1])
        print("Folds: " + str(folds))

    elif sys.argv[i] == "-d" or sys.argv[i] == "--data":
        filename = sys.argv[i+1]
        print("Data file: " + str(filename))

    elif sys.argv[i] == "-s" or sys.argv[i] == "--stepFunction":
        functionName = sys.argv[i+1]
        stepFunction = STEPFUNCTIONS[functionName]
        print("Step Function: " + str(functionName))

print("--------------------------")

trainingSample = helperInstance.readData(filename)

batch = batchLearner()
print("Test reading: " + str(trainingSample[0]))

print("--------------------------")
print("--------------------------")
print("-------STARTING-----------")
print("--------------------------")

batch.optimizeParameters(classifier, PARAMETERS, optimizeID, parRange, trainingSample, startWeights, MAXSTEPS, MAXNONCHANGINGSTEPS, folds, parName, stepFunction)
