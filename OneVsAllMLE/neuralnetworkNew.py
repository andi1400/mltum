__author__ = 'frederik'
from helper import helper
import numpy as np
import copy
import time
import random

class neuralnetworkNew:
    CLASSES = None

    MAX_STEPS = 1000000
    UPDATE_THRESHOLD = None
    MAX_NONCHANGING_STEPS = 100000

    start = None
    endTime = None

    accuracy = []
    maxAccuracyIndex = 0


    #hyper parameters for neural network
    LEARNING_RATE = 1e-4
    SHRINKAGE = 1
    NUM_LAYERS = 5
    NEURONS_PER_LAYER = 16

    BASIS_FUNCTION = helper.getXDirectly
    SIGMOID = helper.sigmoid

    parameterNames = ["Alpha", "SHRINKAGE", "NUM_LAYERS", "NEURONS_PER_LAYER", "BASIS_FUNCTION", "SIGMOID"]
    parameters = None

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
        self.parameters = [self.LEARNING_RATE, self.SHRINKAGE, self.NUM_LAYERS, self.NEURONS_PER_LAYER, self.BASIS_FUNCTION.__name__, self.SIGMOID.__name__]

        self.maxWeights = []
        for k in range(len(self.NEURONS_PER_LAYER)-1):
            self.maxWeights.append(np.ones(shape=(self.NEURONS_PER_LAYER[k]+1, self.NEURONS_PER_LAYER[k+1])))
        for k in range(len(self.NEURONS_PER_LAYER)-1):
            for i in range(len(self.maxWeights[k])):
                for j in range(len(self.maxWeights[k][i])):
                    self.maxWeights[k][i][j] = random.uniform(-1, 1)/(self.NEURONS_PER_LAYER[k+1])**0.5
        print "starting Weights: \n" + str(self.maxWeights) + "\n end"


    def classifySample(self, sample, weights):
        predictedClass = None
        confidenceOfPredicted= None
        classPercentagesNormalized = None

        #let the network run
        lastLayerOutput = self.calcLayerOutputs(sample, weights)[-1][0:len(self.CLASSES)]

        #normalize the output of the last layer to get class percentages
        sumX = sum(lastLayerOutput)
        classPercentagesNormalized = [x/sumX for x in lastLayerOutput]

        confidenceOfPredicted = max(classPercentagesNormalized)
        predictedClass = self.CLASSES[classPercentagesNormalized.index(confidenceOfPredicted)]

        return predictedClass, confidenceOfPredicted, classPercentagesNormalized



    #learning with backpropagation
    def learn(self, trainingSamples, startWeights=None):

        self.start = time.time()
        if (startWeights == None):
            currentWeights = self.maxWeights #in that case, no starting weights have been set.  use the zero-initialized maxWeights as start, then.
        else:
            currentWeights = startWeights
            self.maxWeights = copy.deepcopy(startWeights)
        #print("Cur weights: " + str(currentWeights))
        #Now, learn.
        self.setFilenames()

        for step in range(self.MAX_STEPS):

            reducedLearningRate = self.LEARNING_RATE * self.SHRINKAGE ** step
            for j in range(0, len(trainingSamples)):
                #print str(j) + "/" + str(len(trainingSamples))
                currentWeights = self.learnFromSample(currentWeights, trainingSamples[j], reducedLearningRate)
                #if (j == 0):
                #    print([X[0] for X in self.classifySample(trainingSamples[j][0], currentWeights)[2]])

                #if (j == 0):
                    #for k in range(len(currentWeights)):
                        #print currentWeights[k][0][-1]
                    #    print currentWeights[k]
                    #raw_input()
            #print "weights"
            #for i in range(len(currentWeights)):
            #    print currentWeights[i]
            #raw_input()
            errorBefore, confusionMatrix = self.helper.calcTotalError(self, trainingSamples, currentWeights)
            accuracyStep = 1-errorBefore
            self.accuracy.append(accuracyStep)
            print("Epoch " + str(step) +" Acc: " + str(accuracyStep) + " confusion: " + str(confusionMatrix)) + self.runTime()

            if (accuracyStep > self.accuracy[self.maxAccuracyIndex]):
                self.maxAccuracyIndex = len(self.accuracy) - 1
                self.maxWeights = currentWeights
            #raw_input("Pause")

        #print("Finished learning. Best Accuracy: " + str(bestAccuracy))

    def calcLayerOutputs(self, sample, currentWeights):
        #assign the input.
        outputsPerLayer = []
        sample = copy.deepcopy(sample)
        sample.append(1)
        outputsPerLayer.append(np.array(sample))
        #then propagate forwards.
        #print len(currentWeights)
        #print currentWeights
        for k in range(0, len(currentWeights)): #All the same except for the output layer.
            #print k
            #print len(currentWeights)-1
            if (k == len(currentWeights)-1):
                outputsPerLayer.append(np.ones((self.NEURONS_PER_LAYER[k+1], 1)))
            else:
                outputsPerLayer.append(np.ones((self.NEURONS_PER_LAYER[k+1]+1, 1)))

            for i in range(0, len(currentWeights[k][0])): #do except for the bias:
                #print("_______________________")
                #print currentWeights[k][:, i]

                #print outputsPerLayer[k]
                tmp = np.sum(np.multiply(currentWeights[k][:, i], outputsPerLayer[k]))
                outputsPerLayer[k+1][i] = self.SIGMOID(self.helper, tmp)
        #print "out" + str(outputsPerLayer) + "end"
        #print len(outputsPerLayer)
        #print len(self.maxWeights)
        #raw_input()

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
            #print len(outputsPerLayer[i])
            errorsPerLayer.append(np.zeros((len(outputsPerLayer[i]), 1)))
        for i in range(len(errorsPerLayer[self.NUM_LAYERS-1])):
            if (i < len(self.CLASSES) and sample[1] == self.CLASSES[i]): #In this case, the output should be 1.
                errorsPerLayer[self.NUM_LAYERS-1][i] = (1-outputsPerLayer[self.NUM_LAYERS-1][i])
            else: #In this, 0.
                errorsPerLayer[self.NUM_LAYERS-1][i] = (-outputsPerLayer[self.NUM_LAYERS-1][i])
        #now, it gets funny.: Calculate all of the errors.
        for k in range(len(currentWeights)-1, -1, -1):
            for i in range(len(currentWeights[k])):
                for j in range(0, min(len(errorsPerLayer[k+1]), len(currentWeights[k][i]))):
                    # print "________________________________"
                    # print "i: " + str(i)
                    # print "j: " + str(j)
                    # print currentWeights[k]
                    # print len(errorsPerLayer[k+1])
                    # print errorsPerLayer[k+1]
                    # print errorsPerLayer[k+1][j]
                    # print currentWeights[k][i][j]
                    # print errorsPerLayer[k]
                    #errorsPerLayer[k][i] += np.dot(errorsPerLayer[k+1].transpose(), currentWeights[k][:, i])
                    errorsPerLayer[k][i] += errorsPerLayer[k+1][j] * currentWeights[k][i][j]
        #print "________________________________"
        #print sample
        #print outputsPerLayer
        #print errorsPerLayer
        #raw_input()
        #print "error " + str(errorsPerLayer) + "end"
        #raw_input()
        #Aaaaand calculate the deltaWs.
        deltaW = []
        #First by appending 0-matrices equivalent to currentWeights.
        for k in range(len(currentWeights)):
            deltaW.append(np.zeros((len(currentWeights[k]), len(currentWeights[k][0]))))
            #print "DeltaW: " + str(len(deltaW[k])) + " x " + str(len(deltaW[k][0]))
            #print "c8rW: " + str(len(currentWeights[k])) + " x " + str(len(currentWeights[k][0]))
        #Then, calculate it really. First, the outer as this is a special case.
        #deltaW[-1] = errorsPerLayer[self.NUM_LAYERS].dot(outputsPerLayer[self.NUM_LAYERS-1]))
        for i in range(len(deltaW[-1])):
            for j in range(len(deltaW[-1][i])):
                deltaW[-1][i][j] = errorsPerLayer[-1][j] * outputsPerLayer[-1][j] * (1-outputsPerLayer[-1][j]) * outputsPerLayer[-2][i]
                #deltaW[-1][i][j] = errorsPerLayer[-1][j] * outputsPerLayer[-1][i]
        #print "deltaW: \n" + str(deltaW) + "\n end"
        #And for the rest.
        for k in range(len(currentWeights)-1, -1, -1):
            if (k != len(deltaW)-1):
                #deltaW[k] = np.zeros((self.NEURONS_PER_LAYER[k], self.NEURONS_PER_LAYER[k+1]))
                #TODO MATRIX MULTIPLICATION
                for i in range(0, self.NEURONS_PER_LAYER[k]):
                    for j in range(0, self.NEURONS_PER_LAYER[k+1]):
    #                    print errorsPerLayer[k+1][j]
    #                    print outputsPerLayer[k+1][j]
    #                    print (1-outputsPerLayer[k+1][j])
    #                    print outputsPerLayer[k][i]

                        deltaW[k][i][j] = errorsPerLayer[k+1][i] * outputsPerLayer[k+1][j] * (1-outputsPerLayer[k+1][j]) * outputsPerLayer[k][i]
    #            print deltaW[k]
    #            raw_input()

        #print deltaW
        #raw_input()
        #print "______________________________________-"
        #print ""
        #print "_______________________________________"
        #print "Errors:\n" + str(errorsPerLayer) + "\n end"
        #print "Weights:\n" + str(currentWeights) + "\n end"
        #print "out:\n" + str(outputsPerLayer) + "\n end"
        #print "delta: \n" + str(deltaW) + "\n end"
        modifiedWeights = []

        for k in range(len(currentWeights)):
            modifiedWeights.append(currentWeights[k] + reducedLearningRate * deltaW[k])
        #print modifiedWeights
        #raw_input()
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