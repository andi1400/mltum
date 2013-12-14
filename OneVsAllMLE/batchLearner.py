from helper import helper
import matplotlib.pyplot as plt
import copy
from softZeroOneLoss import softzeroone
from MLEOneVsAll import mleonevsall
from hingeLoss import hinge
import numpy as np
import random


class batchLearner():
    CLASSES = ["sitting", "walking", "standing", "standingup", "sittingdown"]
    helper = None

    def __init__(self):
        self.helper = helper()


    # Range is in [min, max,function, functionparameters)
    def optimizeParameters(self, classifier, parameterStart, optimizeID, parRange, trainingSamples, startWeights, maxSteps, maxNonChangingSteps, folds):
        historicalAccuracy = []
        parameterHistory = []
        accMax = 0
        parMax = parRange[0]
        pOptimize = parRange[0]
        while (pOptimize < parRange[1]):
            parameterHistory.append(pOptimize)
            print("Testing parameter value " + str(pOptimize))
            curParameters = copy.deepcopy(parameterStart)
            curParameters[optimizeID] = pOptimize

            curAcc = self.crossValidate(classifier, curParameters, trainingSamples, startWeights, maxSteps, maxNonChangingSteps, folds)

            historicalAccuracy.append(curAcc)
            if (curAcc > accMax):
                accMax = curAcc
                parMax = pOptimize
            pOptimize, list = parRange[2](pOptimize, parRange[3])

        print("Optimized parameter " + str(optimizeID) + ". Best result: " + str(curAcc) + " for value " + str(parMax))
        plt.plot(parameterHistory, historicalAccuracy)
        return parMax, accMax


    def crossValidate(self, classifier, parameters, trainingSamples, startWeights, maxSteps, maxNonChangingSteps, folds):
        trainingSets = self.splitTrainingSet(trainingSamples, folds)
        listAccs = []
        for i in range(folds):

            classifierInstance = classifier(self.CLASSES, maxSteps, maxNonChangingSteps, parameters)
            classifierInstance.learn(startWeights, self.getCrossValidationSet(trainingSets, i))
            learnedWeights = classifierInstance.maxWeights
            curAcc = 1-self.helper.calcTotalError(classifierInstance, trainingSets[i], learnedWeights)[0]
            listAccs.append(curAcc)
        print listAccs
        return sum(listAccs)/folds

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

def optLogarithmic(p, list):
    if (p >= 10*list[0]):
        list[0] *= 10
    return p + list[0], list

function = optLogarithmic

batch = batchLearner()
helper = helper()
classifier = softzeroone
parameterStart = [1e-5, 0.98, 2, 0]
optimizeID = 1
parRange = [1e-5, 1e-2, function, [1e-5]]
trainingSample = helper.readData("../data/dataset-complete_90PercentTrainingSet_mini10Percent_standardized.arff")
startWeights = []
for i in range(len(CLASSES)):
    dummyWeight = np.zeros(17)
    startWeights.append(dummyWeight)

maxSteps = 1
maxNonChangingSteps = 5

folds = 4

batch.optimizeParameters(classifier, parameterStart, optimizeID, parRange, trainingSample, startWeights, maxSteps, maxNonChangingSteps, folds)