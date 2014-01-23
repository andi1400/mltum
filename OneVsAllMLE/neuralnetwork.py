__author__ = 'frederik'
from helper import helper
import numpy as np
import copy
import time
import random

"""
This class implements a feed forward neural network learning via stochastic back propagation with droput regularization.
"""
class neuralnetwork:
    CLASSES = None

    MAX_STEPS = 1000000
    UPDATE_THRESHOLD = None
    MAX_NONCHANGING_STEPS = 100000

    start = None
    endTime = None

    #Accuracy and accuracyTestSet store the accuracy and the accuracy on the test set respectively, and are storing these values for every step.

    accuracy = []
    maxAccuracyIndex = 0
    accuracyTestSet = []

    #hyper parameters for neural network - these are standard parameters, they will be overwritten in the constructor.
    LEARNING_RATE = 1e-4
    SHRINKAGE = 1
    NUM_LAYERS = 5
    NEURONS_PER_LAYER = 16

    SIGMOID = helper.pseudoSigmoid

    parameterNames = ["Alpha", "SHRINKAGE", "NUM_LAYERS", "NEURONS_PER_LAYER", "SIGMOID"]
    parameters = None

    timeArray = None
    timeLastTime = None

    helper = None

    debugFolderName = None
    weightsFilenameTemplate = None
    confusionFilenameTemplate = None

    #Format for weights is as follows: A list of matrices, one such matrix for each layer.
    #The kth such matrix defines at its [i][j] entry the weight for the connection from the ith neuron of the kth layer to the jth neuron of the k+1th layer.
    #Thereby, the input layer counts as layer 0, and the output layer is therefore the NUM_LAYERS+1th layer.
    maxWeights = None

    defaultWeights = None

    def __init__(self, classes, maxSteps, maxNonChangingSteps, parameters):
        self.helper = helper()
        self.CLASSES = classes
        self.MAX_STEPS = maxSteps
        self.MAX_NONCHANGING_STEPS = maxNonChangingSteps

        self.LEARNING_RATE = parameters[0]
        self.SHRINKAGE = parameters[1]
        self.NUM_LAYERS = int(parameters[2])
        self.NEURONS_PER_LAYER = parameters[3] #NEURONSPERLAYER is a list of the number of neurons that exist per layer - excluding the bias neuron.
        random.seed()
        self.parameters = [self.LEARNING_RATE, self.SHRINKAGE, self.NUM_LAYERS, self.NEURONS_PER_LAYER, self.SIGMOID.__name__]

        self.maxWeights = []
        for k in range(len(self.NEURONS_PER_LAYER)-1):
            self.maxWeights.append(np.ones(shape=(self.NEURONS_PER_LAYER[k]+1, self.NEURONS_PER_LAYER[k+1])))
        for k in range(len(self.NEURONS_PER_LAYER)-1):
            for i in range(len(self.maxWeights[k])):
                for j in range(len(self.maxWeights[k][i])):
                    self.maxWeights[k][i][j] = random.uniform(-1, 1)/(self.NEURONS_PER_LAYER[k+1])**0.5
                    #Starting weights are set randomly, dependant on the number of inputs. Compare lecture 17, neuralnetworks slide 10.

    #classifies a single sample. Uses the weights given. The sample must be a feature vector.
    def classifySample(self, sample, weights):
        #let the network run
        lastLayerOutput = self.calcLayerOutputs(sample, weights)[-1][0:len(self.CLASSES)]

        #normalize the output of the last layer to get class percentages
        sumX = sum(lastLayerOutput)
        classPercentagesNormalized = [x/sumX for x in lastLayerOutput]

        confidenceOfPredicted = max(classPercentagesNormalized)
        predictedClass = self.CLASSES[classPercentagesNormalized.index(confidenceOfPredicted)]

        return predictedClass, confidenceOfPredicted, classPercentagesNormalized



    #learning with backpropagation
    def learn(self, trainingSamples, startWeights=None, testSet=None):
        self.start = time.time()
        if (startWeights == None):
            currentWeights = self.maxWeights #in that case, no starting weights have been set.  use the zero-initialized maxWeights as start, then.
        else:
            currentWeights = startWeights
            self.maxWeights = copy.deepcopy(startWeights)
        #Now, learn.
        self.setFilenames()

        for step in range(self.MAX_STEPS):
            reducedLearningRate = self.LEARNING_RATE * self.SHRINKAGE ** step

            #This implements stochastic back propagation learning.

            #Error calculation and plots.
            errorBefore, confusionMatrix = self.helper.calcTotalError(self, trainingSamples, currentWeights)
            accuracyStep = 1-errorBefore
            self.accuracy.append(accuracyStep)
            print("Epoch " + str(step) + "\t time: " + self.runTime())
            print("\tAcc: " + str("%.4f" % accuracyStep) + " confusion: " + str(confusionMatrix) + " Training")

            if (testSet != None):
                errorBeforeTest, confusionMatrixTest = self.helper.calcTotalError(self, testSet, currentWeights)
                accuracyStepTest = 1-errorBeforeTest

                self.accuracyTestSet.append(accuracyStepTest)

                print("\tAcc: " + str("%.4f" % accuracyStepTest) + " confusion: " + str(confusionMatrixTest) + " Test")
            if (accuracyStep > self.accuracy[self.maxAccuracyIndex]):
                self.maxAccuracyIndex = len(self.accuracy) - 1
                self.maxWeights = currentWeights
            for j in range(0, len(trainingSamples)):
                currentWeights = self.learnFromSample(currentWeights, trainingSamples[j], reducedLearningRate)


    #Calculates the outputs for all layers. This is a list of vectors, with [0] representing the input layer, [1] the first hidden layer and so on.
    def calcLayerOutputs(self, sample, currentWeights):
        #assign the input.
        outputsPerLayer = []
        sample = copy.deepcopy(sample)
        sample.append(1)
        outputsPerLayer.append(np.matrix(sample).transpose())
        #then propagate forwards.

        for k in range(0, len(currentWeights)): #All the same except for the output layer.
            if (k == len(currentWeights)-1):
                outputsPerLayer.append(self.SIGMOID(self.helper, np.dot(currentWeights[k].transpose(), outputsPerLayer[k])))
            else:
                outputsPerLayer.append(np.ones((self.NEURONS_PER_LAYER[k+1]+1, 1)))
                outputsPerLayer[k+1][:-1] = self.SIGMOID(self.helper, np.dot(currentWeights[k].transpose(), outputsPerLayer[k]))
        return outputsPerLayer


    def learnFromSample(self, currentWeights, sample, reducedLearningRate):
        #This is a list of vectors, with the kth entry in the list representing the following:
        #its ith entry is the output of the ith Neuron in the kth layer. Layers as defined under self.maxWeights.
        outputsPerLayer = self.calcLayerOutputs(sample[0], currentWeights)

        #Defined equivalent to outputsPerLayer.
        errorsPerLayer = []

        #afterwards, it will be necessary to get the error of the last layer first. But, before, set the right size for the error.
        #This actually introduces an error for the bias, which we just won't care about.
        for i in range(len(outputsPerLayer)):
            errorsPerLayer.append(np.zeros((outputsPerLayer[i].shape[0], 1)))

        #Set the error for the output layer.
        for i in range(len(errorsPerLayer[self.NUM_LAYERS-1])):
            if (i < len(self.CLASSES) and sample[1] == self.CLASSES[i]): #In this case, the output should be 1.
                errorsPerLayer[self.NUM_LAYERS-1][i] = (1-outputsPerLayer[self.NUM_LAYERS-1][i])
            else: #In this, 0.
                errorsPerLayer[self.NUM_LAYERS-1][i] = (-outputsPerLayer[self.NUM_LAYERS-1][i])

        #now, it gets funny.: Calculate all of the errors.
        for k in range(len(currentWeights)-1, -1, -1):
            if (k == len(currentWeights)-1):
                errorsPerLayer[k] = np.dot(currentWeights[k], errorsPerLayer[k+1])
            else:
                errorsPerLayer[k] = np.dot(currentWeights[k], errorsPerLayer[k+1][0:-1])

        #And calculate the deltaWs.
        deltaW = []
        for k in range(len(currentWeights)):
            deltaW.append(np.zeros(shape=currentWeights[k].shape))
        for k in range(len(currentWeights)-1, -1, -1):
            if (k == len(currentWeights)-1):
                tmp = np.multiply(np.multiply(errorsPerLayer[k+1], outputsPerLayer[k+1]), 1-outputsPerLayer[k+1]).transpose()
            else:
                tmp = (np.multiply(np.multiply(errorsPerLayer[k+1], outputsPerLayer[k+1]), 1-outputsPerLayer[k+1]))[0:-1].transpose()
            deltaW[k] = np.dot(outputsPerLayer[k], tmp)
        modifiedWeights = []
        for k in range(len(currentWeights)):
            modifiedWeights.append(currentWeights[k] + reducedLearningRate * deltaW[k])
        return modifiedWeights


    def setFilenames(self):
        self.debugFolderName = "../output/weights/debug/" + str(self.start) + "_" + str(self.__class__.__name__) + "/"
        self.weightsFilenameTemplate = self.debugFolderName + str(self.start)
        self.confusionFilenameTemplate = self.debugFolderName + str(self.start)

        print("Writing DEBUG Information to " + str(self.debugFolderName) + "...")

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

    def runTime(self):
        return str(int(time.time() - self.getStartTime()))