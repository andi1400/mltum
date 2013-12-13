import time
import numpy as np
import math
from helper import helper
import copy

class mleonevsall():
    CLASSES = None

    MAX_STEPS = 10000000
    UPDATE_THRESHOLD = None
    MAX_NONCHANGING_STEPS = 100000

    start = None
    endTime = None

    accuracy = []
    maxAccuracyIndex = 0
    maxWeights = None

    #hyper parameters for soft zero one loss
    LEARNING_RATE = 1e-4
    SHRINKAGE = 0.99

    BASIS_FUNCTION = helper.getXDirectly
    SIGMOID = helper.pseudoSigmoid
    parameterNames = ["Alpha", "Shrinkage", "BASIS_FUNCTION", "SIGMOID"]
    parameters = [LEARNING_RATE, SHRINKAGE, BASIS_FUNCTION, SIGMOID]

    helper = None

    def __init__(self, classes):
        self.helper = helper()
        self.CLASSES = classes

    def classifySample(self, x, ClassWeights):
        classPercentages = np.zeros(len(self.CLASSES))

        for i in range(len(self.CLASSES)):
            currentWeightVector = ClassWeights[i]

            classPercentages[i] = self.classifySampleSingleClass(x, currentWeightVector)[1]

        sumX = sum(classPercentages)
        classPercentagesNormalized = [x/sumX for x in classPercentages]
        confidenceOfPredicted = max(classPercentagesNormalized)
        predictedClass = self.CLASSES[classPercentagesNormalized.index(confidenceOfPredicted)]

        return predictedClass, confidenceOfPredicted, classPercentages


    #Will not do oneVsAll but perform ONE logistic regression classification.
    #returns class(1: right class, 0:wrong class) and confidence
    def classifySampleSingleClass(self, x, ClassWeight):
        currentFeatureVector = self.helper.getPhi(x, self.BASIS_FUNCTION)

        #print "currentFeatureVec" + str(currentFeatureVector)
        #print "ClassWeights" + str(ClassWeight)
        wTimesPhi = np.dot(np.transpose(ClassWeight), currentFeatureVector)
        #print "wTimesPhi" + str(wTimesPhi)
        regressionResult = self.SIGMOID(self.helper, wTimesPhi)

        if(regressionResult >= 0.5):
            return 1, regressionResult

        return 0, regressionResult


    def learn(self, startWeights, trainingSamples):
        self.start = time.time()
        curWeights = copy.deepcopy(startWeights)
        self.maxWeights = copy.deepcopy(startWeights)

        for i in range(self.MAX_STEPS):
            curWeights = self.optimizeAllWeights(curWeights, trainingSamples, i)

            if(i % 10 == 0):
                self.helper.writeWeightsDebug("../output/weights/debug/" + str(self.start) + "_step" + str(i) + ".csv", curWeights)

            #print(curWeights)
            #termination check on no improvement
            if(i - self.maxAccuracyIndex >= self.MAX_NONCHANGING_STEPS and self.maxWeights != None):
                break


    #Will optimize all the weights for every class. Thereby it does one step for every class and then contiues to the next step.
    def optimizeAllWeights(self, currentWeights, trainingSamples, step):

        for c in range(len(self.CLASSES)):
            currentWeights[c] = self.updateWeightsPerClasStep(currentWeights[c], trainingSamples, self.CLASSES[c], self.LEARNING_RATE * (self.SHRINKAGE ** step))


        #check the current error and compute accuracy, then do a debug output to see the progress
        currentGeneralError = self.helper.calcTotalError(self, trainingSamples, currentWeights)[0]
        currentAccuracy = 1- currentGeneralError
        print("Progress Global Weight: " + str(step) + " Right: " + str(1-currentGeneralError) + self.helper.strRuntime(self.start))
        self.accuracy.append(currentAccuracy) #save accuracy for later result printing

        #check if we need to store the new accuracy as the new best one
        if(currentAccuracy > self.accuracy[self.maxAccuracyIndex]):
            self.maxAccuracyIndex = step
            self.maxWeights = currentWeights

        self.endTime = time.time()

        return currentWeights


    #Will optimize the weights for one class only. Thereby this will only do one step of gradient decent.
    #CurrentWeightsPerClass is the vector contining the weights for this class logistic regression. Training Samples is a list of training samples. Current Class is nominal (string) class value.
    def updateWeightsPerClasStep(self, currentWeightsPerClass, trainingSamples, currentClass, shrinkedLearningRate):
        newWeights = currentWeightsPerClass
        deltaW = np.zeros(len(currentWeightsPerClass))

        for sample in trainingSamples:
            sampleInput = sample[0]
            sampleTarget = sample[1]
            prediction = self.classifySampleSingleClass(sampleInput, newWeights)

            target = 0
            if sampleTarget == currentClass:
                target = 1
            for j in range(len(currentWeightsPerClass)):
                deltaW[j] += (target - prediction[0]) * self.BASIS_FUNCTION(self.helper, sampleInput, j)

        #update w with learning rate of its gradient.
        #change1 weights can only be updated with complete gradient
        newWeights = newWeights + deltaW * shrinkedLearningRate

        return newWeights

    def getAccuracy(self):
        return self.accuracy

    def getWeights(self):
        return self.maxWeights

    def getParameterNameList(self):
        return self.parameterNames

    def getParameterList(self):
        return self.parameters

    def getStartTime(self):
        return self.start