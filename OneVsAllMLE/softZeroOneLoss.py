import time
import numpy as np
import copy
import math
from helper import helper

class softzeroone():
    CLASSES = None

    MAX_STEPS = 1000000
    UPDATE_THRESHOLD = None
    MAX_NONCHANGING_STEPS = 100000

    start = None
    endTime = None

    accuracy = []
    maxAccuracyIndex = 0
    maxWeights = None

    #hyper parameters for soft zero one loss
    LEARNING_RATE = 1e-0
    SHRINKAGE = 1
    BETA = 2
    REGULARIZER = 0 #lambda
    GRADIENTSTEPSIZE = 0.1
    BASIS_FUNCTION = helper.getXDirectly
    SIGMOID = helper.pseudoSigmoid
    parameterNames = ["Alpha", "Shrinkage", "Beta", "Lambda", "H", "BASIS_FUNCTION", "SIGMOID"]
    parameters = [LEARNING_RATE, SHRINKAGE, BETA, REGULARIZER, GRADIENTSTEPSIZE, BASIS_FUNCTION, SIGMOID]

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

        wTimesPhi = np.dot(np.transpose(ClassWeight), currentFeatureVector)

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

            print(curWeights)
            #termination check on no improvement
            if(i - self.maxAccuracyIndex >= self.MAX_NONCHANGING_STEPS and self.maxWeights != None):
                break


    #Will optimize all the weights for every class. Thereby it does one step for every class and then contiues to the next step.
    def optimizeAllWeights(self, currentWeights, trainingSamples, step):
        tempWeightsOld = currentWeights

        for c in range(len(self.CLASSES)):
            currentWeights[c] = self.updateWeightsPerClasStep(tempWeightsOld[c], trainingSamples, self.CLASSES[c], self.LEARNING_RATE * (self.SHRINKAGE ** step))


        #check the current error and compute accuracy, then do a debug output to see the progress
        currentGeneralError = self.helper.calcTotalError(self, trainingSamples)
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

            #Done to get from  multiclass to binary target. 0 is not class, 1 is class.
            target = 0
            if sampleTarget == currentClass:
                target = 1

            #Use the h method to get the gradient in all dimensions
            weightsPlusH = copy.deepcopy(newWeights)
            regParameter = float(self.REGULARIZER)/len(trainingSamples)
            errorOriginal = self.calcError(target, sampleInput, newWeights, regParameter)


            for j in range(len(currentWeightsPerClass)):
                weightsPlusH[j] += self.GRADIENTSTEPSIZE
                errorNew = self.calcError(target, sampleInput, weightsPlusH, regParameter)
                deltaW[j] += (errorOriginal - errorNew)/self.GRADIENTSTEPSIZE #TODO Minus right?

                weightsPlusH[j] -= self.GRADIENTSTEPSIZE
        #update w with learning rate of its gradient.
        #change1 weights can only be updated with complete gradient
        newWeights = newWeights + deltaW * shrinkedLearningRate

        return newWeights

    #calculates the error for one input, the one with the result target.
    def calcError(self, target, inputVector, weights, regParameter):
        basis = self.BASIS_FUNCTION
        currentFeatureVector = self.helper.getPhi(inputVector, basis)

        #calc (sigmoid(BETA* (w * phi) - target))^2 + lambda * w^2
        wTimesPhi = np.dot(np.transpose(weights), currentFeatureVector)
        result = (self.SIGMOID(self.helper, self.BETA * wTimesPhi) - target)**2
        result += regParameter * np.dot(np.transpose(weights), weights)

        return result

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
