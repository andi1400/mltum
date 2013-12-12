import time
import numpy as np
import math

class mleonevsall():
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

    def __init__(self, classes, learningrate, shrinkage, maxsteps, maxstepsnochange, updatethreshold, starttime, classifier):
        self.CLASSES = classes
        self.LEARNING_RATE = learningrate
        self.SHRINKAGE = shrinkage
        self.MAX_STEPS = maxsteps
        self.MAX_NONCHANGING_STEPS = maxstepsnochange
        self.UPDATE_THRESHOLD = updatethreshold
        self.start = starttime
        self.classifier = classifier



    #Will optimize all the weights for every class. Thereby it does one step for every class and then contiues to the next step.
    def optimizeAllWeights(self, currentWeights, trainingSamples, step):
        tempWeightsOld = currentWeights

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
            prediction = self.classifier.classifySampleSingleClass(sampleInput, newWeights)
            target = 0
            if sampleTarget == currentClass:
                target = 1
            for j in range(len(currentWeightsPerClass)):
                #deltaW[j] += abs(prediction[0] - target) * prediction[1] * getBasisOutput(sampleInput, j)
                deltaW[j] += (target - prediction[0]) * self.classifier.getBasisOutput(sampleInput, j)

        #update w with learning rate of its gradient.
        #change1 weights can only be updated with complete gradient
        newWeights = newWeights + deltaW * shrinkedLearningRate

        return newWeights


    def getAllFitnesses(self):
        return self.fitness