from helper import helper
import numpy as np
import copy
import time
import random

class neuralnetwork:
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

    #weight list containing one list per layer. Each of these inner lists themselfs contains one list per neuron
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
        self.NEURONS_PER_LAYER = int(parameters[3])
        random.seed()
        self.parameters = [self.LEARNING_RATE, self.SHRINKAGE, self.NUM_LAYERS, self.NEURONS_PER_LAYER, self.BASIS_FUNCTION.__name__, self.SIGMOID.__name__]

        self.maxWeights = []
        for i in range(self.NUM_LAYERS):
            self.maxWeights.append([])

            for j in range(self.NEURONS_PER_LAYER):
                self.maxWeights[i].append([])

                #now for each of the neurons from last layer to the next + some bias value as input to each neuron
                for k in range(self.NEURONS_PER_LAYER+1):
                    self.maxWeights[i][j].append(0)

        self.defaultWeights = []
        for i in range(self.NUM_LAYERS):
            self.defaultWeights.append([])

            for j in range(self.NEURONS_PER_LAYER):
                self.defaultWeights[i].append([])

                #now for each of the neurons from last layer to the next + some bias value as input to each neuron
                for k in range(self.NEURONS_PER_LAYER+1):
                    self.defaultWeights[i][j].append(random.random()*2 -1)

        #print("Initialized Neurol Network weights to one:")
        #print(self.defaultWeights)

    def classifySample(self, inputVector, ClassWeights):
        predictedClass = None
        confidenceOfPredicted= None
        classPercentagesNormalized = None

        #let the network run
        lastLayerOutput = self.calcFinalOutput(inputVector, ClassWeights)[0]

        #normalize the output of the last layer to get class percentages
        sumX = sum(lastLayerOutput)
        classPercentagesNormalized = [x/sumX for x in lastLayerOutput]

        confidenceOfPredicted = max(classPercentagesNormalized)
        predictedClass = self.CLASSES[classPercentagesNormalized.index(confidenceOfPredicted)]

        return predictedClass, confidenceOfPredicted, classPercentagesNormalized


    def calcSingleNeuronInput(self, neuronId, inputFromBefore, neuronWeights):
        #on the first step add the bias input for this neuron which is the first weight
        netInputSingleNeuron = neuronWeights[0]
        for j in range(self.NEURONS_PER_LAYER):
            #iteratively sum up \sum_allInputWeights (x_i \cdot w_{i,j})
            netInputSingleNeuron += inputFromBefore[j] * neuronWeights[j + 1] #j+1 because of basis weight

        return netInputSingleNeuron

    def calcLayerOutput(self, inputFromBefore, layerId, layerWeights):
        #every member is the output for one neuron in this layer.
        #it will be summed up iteratively
        outputSums = []

        for i in range(self.NEURONS_PER_LAYER):
            netInputSingleNeuron = self.calcSingleNeuronInput(i, inputFromBefore, layerWeights[i])

            #do a sigmpoid of it and append it as this neurons output
            outputSingleNeuron = self.SIGMOID(self.helper, netInputSingleNeuron)
            outputSums.append(outputSingleNeuron)

        return outputSums


    def calcFinalOutput(self, inputFromBefore, currentWeights):
        lastLayerOutput = inputFromBefore
        outputsPerLayer = []

        for currentLayer in range(self.NUM_LAYERS):
            layerWeights = currentWeights[currentLayer]
            lastLayerOutput = self.calcLayerOutput(lastLayerOutput, currentLayer, layerWeights)
            outputsPerLayer.append(lastLayerOutput)

        return lastLayerOutput[0: len(self.CLASSES)], outputsPerLayer


    #learning with backpropagation
    def learn(self, trainingSamples, startWeights=None):
        if (startWeights == None):
            startWeights = self.defaultWeights
        #measure the start ime
        self.start = time.time()
        #set the debug filenames and create folders
        self.setFilenames()



        currentWeights = startWeights
        bestAccuracy = 0

        for step in range(self.MAX_STEPS):
            #random.shuffle(trainingSamples)
            reducedLearningRate = self.LEARNING_RATE * self.SHRINKAGE ** step
            currentWeights, accuracyStep, confusionMatrix = self.performLearnStep(currentWeights, trainingSamples, reducedLearningRate)
            #print("Weights:")
            #print(len(currentWeights[-1][0]))
            percentageMax_steps = self.MAX_STEPS/100
            #if(step % percentageMax_steps == 0):
            if(step % 10 == 0):
                self.helper.writeWeightsDebug(self.weightsFilenameTemplate + "_step" + str(step) + ".csv", currentWeights)
#               print ("Finished " + str(step/percentageMax_steps) + "%.")
            self.accuracy.append(accuracyStep)
            print("Accuracy step " + str(step) +": " + str(accuracyStep) + " confusion: " + str(confusionMatrix)) + self.runTime()
            if(accuracyStep > bestAccuracy):
                #print("Found better accuracy: " + str(accuracyStep))
                bestAccuracy = accuracyStep
                self.maxWeights = currentWeights


        #print("Finished learning. Best Accuracy: " + str(bestAccuracy))


    #will do one step of learning. E.g. go from the end of the network to the front for every sample.
    def performLearnStep(self, currentWeights, trainingSamples, reducedLearningRate):
        for sID in range(len(trainingSamples)):
            #if (sID % 1000 == 0):
            #    print sID
            currentWeights = self.learnFromSample(currentWeights, trainingSamples[sID], reducedLearningRate)

        errorAfterStep, confusionMatrix = self.helper.calcTotalError(self, trainingSamples, currentWeights)

        accuracyAfterStep = 1- errorAfterStep

        return currentWeights, accuracyAfterStep, confusionMatrix

    def learnFromSample(self, currentWeights, sample, reducedLearningRate):
        lastError = None
        #modifiedWeights = copy.deepcopy(currentWeights)
        modifiedWeights = []
        ## append the list for each layer.
        for i in range(len(currentWeights)):
            modifiedWeights.append([])
        #first do a forward run to get all the outputs on each layer
        finalOutput, outputsPerLayer = self.calcFinalOutput(sample[0], currentWeights)

        #at the first iteration the last error is different than the inner ones
        #returns the error \delta_j for every neuron.
        lastError = self.getOutputError(outputsPerLayer, currentWeights[-1], sample[1])

        #modifiy weights for the output layer
        deltaW = self.calcDeltaWPerLayer(lastError, outputsPerLayer[-1], reducedLearningRate)

        #w_old + deltaW
        for i in range(len(currentWeights[-1])):
            modifiedWeights[-1].append([])
            for j in range(len(currentWeights[-1][i])):
                modifiedWeights[-1][i].append(currentWeights[-1][i][j] + deltaW[i][j])

        for layerId in range(self.NUM_LAYERS - 2, -1, -1):
            #calculate that layer's error - now different from the output error
            #needs the deltaKs as parameters
            #TODO test if layerID -1 or +1 for outputsperLayer
            lastError = self.getInnerError(outputsPerLayer[layerId - 1], lastError, currentWeights[layerId], currentWeights[layerId + 1])

            # modify weights for the layer
            deltaW = self.calcDeltaWPerLayer(lastError, outputsPerLayer[layerId -1], reducedLearningRate)

            #w_old + deltaW
            for i in range(len(currentWeights[layerId])):
                modifiedWeights[layerId].append([])
                for j in range(len(currentWeights[layerId][i])):
                    modifiedWeights[layerId][i].append(currentWeights[layerId][i][j] + deltaW[i][j])

        return modifiedWeights


    def getOutputError(self, outputsPerLayer, thisLayerWeights, target):
        errors = []

        for i in range(self.NEURONS_PER_LAYER):
            #for the ith neuron, see which the correct target is. So
            #for all neurons the target will be 0, except the for the
            #output neuron that shall predict for this class.

            targetValue = 0
            if i == self.CLASSES.index(target):
                targetValue = 1

            targetMinusOutput = targetValue - outputsPerLayer[-1][i]

            inputFromLastLayer = self.calcSingleNeuronInput(i, outputsPerLayer[-2], thisLayerWeights[i])
            sigmoidDerivative = self.SIGMOID(self.helper, inputFromLastLayer) * (1-self.SIGMOID(self.helper, inputFromLastLayer))
            errorSignal = sigmoidDerivative * targetMinusOutput

            errors.append(errorSignal)

        return errors

    def getInnerError(self, beforeLayerOutput, lastError, thisLayerWeights, nextLayerWeights):
        errors = []

        for i in range(self.NEURONS_PER_LAYER):
            #calculate the error as the weighted errors of all successors
            summedWeightedError = 0
            for j in range(len(lastError)):
                #summedWeightedError += lastError[j] * nextLayerWeights[j][i+1]
                summedWeightedError += lastError[j] * nextLayerWeights[i][j+1]


            #net_j
            inputFromLastLayer = self.calcSingleNeuronInput(i, beforeLayerOutput, thisLayerWeights[i])

            sigmoidDerivative = self.SIGMOID(self.helper, inputFromLastLayer) * (1-self.SIGMOID(self.helper, inputFromLastLayer))
            errorSignal = sigmoidDerivative * summedWeightedError

            errors.append(errorSignal)
        return errors


    #now we optimize backwards each weight w (think of an arrow that goes into this neuron). So we have
    #as many arriving weights as neurons per layer +  the bias weight
    def calcDeltaWPerNeuron(self, neuronError, lastLayerOutput, learningRate):
        result = []

        #treat the bias as special case
        result.append(learningRate * neuronError * 1)

        for i in range(self.NEURONS_PER_LAYER):
            result.append(learningRate * neuronError * lastLayerOutput[i])

        return result

    def calcDeltaWPerLayer(self, lastLayerError, lastLayerOutput, learningRate):
        deltaWList = []

        for i in range(self.NEURONS_PER_LAYER):
            deltaWList.append(self.calcDeltaWPerNeuron(lastLayerError[i], lastLayerOutput, learningRate))

        return deltaWList


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