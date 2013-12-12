import time
import numpy as np
import copy
import math

class softzeroone():
    CLASSES = None
    LEARNING_RATE = None
    SHRINKAGE = None

    MAX_STEPS = None
    UPDATE_THRESHOLD = None
    MAX_NONCHANGING_STEPS = None

    start = None

    fitness = []
    maxFitnessIndex = 0
    maxWeights = None

    classifier = None

    #hyper parameters for soft zero one loss
    beta = None
    regularizer = None
    GRADIENTSTEPSIZE = None

    def __init__(self, classes, learningrate, shrinkage, maxsteps, maxstepsnochange, updatethreshold, starttime, classifier, beta, regularizer, gradientstepsize):
        self.CLASSES = classes
        self.LEARNING_RATE = learningrate
        self.SHRINKAGE = shrinkage
        self.MAX_STEPS = maxsteps
        self.MAX_NONCHANGING_STEPS = maxstepsnochange
        self.UPDATE_THRESHOLD = updatethreshold
        self.start = starttime
        self.classifier = classifier
        self.beta = beta
        self.regularizer = regularizer
        self.GRADIENTSTEPSIZE = gradientstepsize


    #Will optimize all the weights for every class. Thereby it does one step for every class and then contiues to the next step.
    def optimizeAllWeights(self, currentWeights, trainingSamples, step):
        tempWeightsOld = currentWeights
        if (self.maxWeights == None):
            self.maxWeights = copy.deepcopy(currentWeights)
        tempWeightsOld = currentWeights
        for c in range(len(self.CLASSES)):
            currentWeights[c] = self.updateWeightsPerClasStep(tempWeightsOld[c], trainingSamples, self.CLASSES[c], self.LEARNING_RATE * (self.SHRINKAGE ** step))

        currentGeneralError = self.classifier.calcTotalError(currentWeights, trainingSamples)
        currentAccuracy = 1- currentGeneralError

        print("Progress Global Weight: " + str(step) + " Right: " + str(1-currentGeneralError) + self.classifier.runtime())
        self.fitness.append(currentAccuracy)

        if(currentAccuracy > self.fitness[self.maxFitnessIndex]):
            self.maxFitnessIndex = step
            self.maxWeights = currentWeights

        if(step - self.maxFitnessIndex >= self.MAX_NONCHANGING_STEPS and self.maxWeights != None):
            return self.maxWeights

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
            regParameter = float(self.regularizer)/len(trainingSamples)
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
        currentFeatureVector = self.classifier.getPhi(inputVector)

        wTimesPhi = np.dot(np.transpose(weights), currentFeatureVector)

        result = (self.classifier.sigmoid(self.beta * wTimesPhi) - target)**2

        result += regParameter * np.dot(np.transpose(weights), weights)

        return result

    def getAllFitnesses(self):
        return self.fitness