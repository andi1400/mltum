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
    SIGMOID = helper.pseudoSigmoid

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
        self.NEURONS_PER_LAYER = int(parameters[3]) #NEURONSPERLAYER is a list of the number of neurons that exist per layer - excluding the bias neuron.
        random.seed()
        self.parameters = [self.LEARNING_RATE, self.SHRINKAGE, self.NUM_LAYERS, self.NEURONS_PER_LAYER, self.BASIS_FUNCTION.__name__, self.SIGMOID.__name__]

        self.maxWeights = []
        for k in range(self.NUM_LAYERS):
            self.maxWeights.append(np.ones(shape=(self.NEURONS_PER_LAYER[k]+1, self.NEURONS_PER_LAYER[k+1])))

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
        for i in range(self.MAX_STEPS):

            reducedLearningRate = self.LEARNING_RATE * self.SHRINKAGE ** i
            for j in range(0, len(trainingSamples)):
                currentWeights = self.learnFromSample(currentWeights, trainingSamples[j], reducedLearningRate)
                if (j == 0):
                    for k in range(len(currentWeights)):
                        print currentWeights[k][1][1]

            errorBefore, confusionMatrix = self.helper.calcTotalError(self, trainingSamples, currentWeights)
            accuracyStep = 1-errorBefore
            self.accuracy.append(accuracyStep)
            print("Epoch " + str(i) +" Acc: " + str(accuracyStep) + " confusion: " + str(confusionMatrix)) + self.runTime()

            if (accuracyStep > self.accuracy[self.maxAccuracyIndex]):
                self.maxAccuracyIndex = len(self.accuracy) - 1
                self.maxWeights = currentWeights
            #raw_input("Pause")

        #print("Finished learning. Best Accuracy: " + str(bestAccuracy))

    def calcLayerOutputs(self, sample, currentWeights):
        #assign the input.
        outputsPerLayer = []
        sample.append(1)
        outputsPerLayer.append(np.array(sample))

        #then propagate forwards.
        for k in range(0, len(currentWeights)):
            outputsPerLayer.append(np.ones((self.NEURONS_PER_LAYER, 1)))
            for i in range(0, len(outputsPerLayer[k])-1):
                tmp = np.sum(np.multiply(currentWeights[k][i], outputsPerLayer[k]))
                outputsPerLayer[k+1][i] = self.SIGMOID(self.helper, tmp)
            outputsPerLayer[k+1][-1] = 1
        return outputsPerLayer


    def learnFromSample(self, currentWeights, sample, reducedLearningRate):
        #This is a list of vectors, with the kth entry in the list representing the following:
        #its ith entry is the output of the ith Neuron in the kth layer. Layers as defined under self.maxWeights.
        outputsPerLayer = self.calcLayerOutputs(sample[0], currentWeights)

        #Defined equivalent to outputsPerLayer.
        errorsPerLayer = []
        #afterwards, it will be necessary to get the error of the last layer first. But, before, set the right size for the error.
        for i in range(len(outputsPerLayer)):
            errorsPerLayer.append(np.zeros((1, len(outputsPerLayer[i]))))

        errorsPerLayer[self.NUM_LAYERS] = np.zeros((len(outputsPerLayer[self.NUM_LAYERS])+1, 1)) #set all to 0, first.
        for i in range(len(errorsPerLayer[self.NUM_LAYERS])):
            if (i < len(self.CLASSES) and sample[1] == self.CLASSES[i]): #In this case, the output should be 1.
                errorsPerLayer[self.NUM_LAYERS][i] = (1-outputsPerLayer[self.NUM_LAYERS][i])**2
            else: #In this, 0.
                if (i == len(errorsPerLayer[self.NUM_LAYERS])) - 1:
                    errorsPerLayer[self.NUM_LAYERS][i] == 0 #BIAS
                else:
                    errorsPerLayer[self.NUM_LAYERS][i] = (outputsPerLayer[self.NUM_LAYERS][i])**2
        #now, it gets funny.: Calculate all of the errors.
        for k in range(len(errorsPerLayer)-2, -1, -1):
            for i in range(len(errorsPerLayer[k])):
                print errorsPerLayer[k+1]
                print currentWeights[k][i]
                #errorsPerLayer[k] = (errorsPerLayer[k+1].dot(currentWeights[k][i]))
                errorsPerLayer[k][i] = np.dot(errorsPerLayer[k+1].transpose(), currentWeights[k][i])
                print errorsPerLayer[k]
                raw_input("_")
        #print "Errors per layer\n" + str(errorsPerLayer) + "\n finished."
        #raw_input("_")
        #Aaaaand calculate the deltaWs.
        deltaW = []
        #First by appending 0-matrices equivalent to currentWeights.
        for k in range(len(currentWeights)):
            deltaW.append(np.zeros((len(currentWeights[k]), len(currentWeights[k][0]))))
        #Then, calculate it really. First, the outer as this is a special case.
        #deltaW[-1] = errorsPerLayer[self.NUM_LAYERS].dot(outputsPerLayer[self.NUM_LAYERS-1])
        deltaW[-1] = outputsPerLayer[self.NUM_LAYERS-1].dot(errorsPerLayer[self.NUM_LAYERS].transpose()).dot(1-outputsPerLayer[self.NUM_LAYERS-1])
        #And for the rest.
        for k in range(len(currentWeights)-1, 0, -1):
            deltaW[k] = np.zeros((self.NEURONS_PER_LAYER, self.NEURONS_PER_LAYER))
            #TODO MATRIX MULTIPLICATION
            for i in range(0, self.NEURONS_PER_LAYER):
                for j in range(0, self.NEURONS_PER_LAYER):
                    print errorsPerLayer[k][i]
                    print outputsPerLayer[k][j][0]
                    print outputsPerLayer[k-1][i]
                    deltaW[k][i][j] = errorsPerLayer[k][i] * outputsPerLayer[k][j][0] * (1- outputsPerLayer[k][j][0]) * outputsPerLayer[k-1][i]
        #print deltaW
        #raw_input("")
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
        return str(time.time() - self.getStartTime())