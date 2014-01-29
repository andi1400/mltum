import time
import numpy as np
import copy
import math
from helper import helper


"""
This class implements a hinge loss learning logistic regression classifier.
"""
class hinge():
    CLASSES = None

    MAX_STEPS = 10000
    UPDATE_THRESHOLD = None
    MAX_NONCHANGING_STEPS = 100000

    start = None
    endTime = None

    accuracy = []
    accuracyTestSet = []
    maxAccuracyIndex = 0
    maxWeights = None

    defaultWeights = None

    #hyper parameters for soft zero one loss
    LEARNING_RATE = 1e-2
    SHRINKAGE = 0.98

    BASIS_FUNCTION = helper.getXDirectly
    SIGMOID = helper.pseudoSigmoid
    parameterNames = ["Alpha", "Shrinkage", "BASIS_FUNCTION", "SIGMOID"]
    parameters = [LEARNING_RATE, SHRINKAGE, BASIS_FUNCTION.__name__, SIGMOID.__name__]

    debugFolderName = None
    weightsFilenameTemplate = None
    confusionFilenameTemplate = None

    helper = None

    def __init__(self, classes, maxSteps, maxNonChangingSteps, parametersGiven):
        self.helper = helper()
        self.CLASSES = classes
        self.MAX_STEPS = maxSteps
        self.MAX_NONCHANGING_STEPS = maxNonChangingSteps
        self.LEARNING_RATE = parametersGiven[0]
        self.SHRINKAGE = parametersGiven[1]

        print("parameters given: " + str(parametersGiven))

        #resetting parameters for correct displaying
        self.parameters = [self.LEARNING_RATE, self.SHRINKAGE, self.BASIS_FUNCTION.__name__, self.SIGMOID.__name__]

        self.maxWeights = []
        for i in range(len(classes)):
            dummyWeight = np.zeros(17)
            self.maxWeights.append(dummyWeight)

        self.defaultWeights = []
        for i in range(len(self.CLASSES)):
            dummyWeight = np.zeros(17)
            self.defaultWeights.append(dummyWeight)

    def classifySample(self, x, ClassWeights):
        classPercentages = np.zeros(len(self.CLASSES))
        for i in range(len(self.CLASSES)):
            currentWeightVector = ClassWeights[i]

            classPercentages[i] = self.classifySampleSingleClass(x, currentWeightVector)[1]

        sumX = sum(classPercentages)
        classPercentagesNormalized = [x/sumX for x in classPercentages]
        confidenceOfPredicted = max(classPercentagesNormalized)
        predictedClass = self.CLASSES[classPercentagesNormalized.index(confidenceOfPredicted)]

        return predictedClass, confidenceOfPredicted, classPercentagesNormalized


    #Will not do oneVsAll but perform ONE logistic regression classification.
    #returns class(1: right class, 0:wrong class) and confidence
    def classifySampleSingleClass(self, x, ClassWeight):
        currentFeatureVector = self.helper.getPhi(x, self.BASIS_FUNCTION)

        wTimesPhi = np.dot(np.transpose(ClassWeight), currentFeatureVector)

        regressionResult = self.SIGMOID(self.helper, wTimesPhi)

        if(regressionResult >= 0.5):
            return 1, regressionResult

        return 0, regressionResult

    def learn(self, trainingSamples):
        self.learn(self.defaultWeights, trainingSamples)

    def learn(self, trainingSamples, startWeights=None, testSet=None):
        if (startWeights is None):
            startWeights = self.defaultWeights
        #measure the start ime
        self.start = time.time()
        #set the debug filenames and create folders
        self.setFilenames()


        curWeights = copy.deepcopy(startWeights)
        self.maxWeights = copy.deepcopy(startWeights)

        for i in range(self.MAX_STEPS):
            curWeights = self.optimizeAllWeights(curWeights, trainingSamples, i, testSet)

            if(i % 10 == 0):
                self.helper.writeWeightsDebug(self.weightsFilenameTemplate + "_step" + str(i) + ".csv", self.maxWeights)

            #termination check on no improvement
            if(i - self.maxAccuracyIndex >= self.MAX_NONCHANGING_STEPS and self.maxWeights != None):
                break


    #Will optimize all the weights for every class. Thereby it does one step for every class and then contiues to the next step.
    def optimizeAllWeights(self, currentWeights, trainingSamples, step, testSet = None):
        for c in range(len(self.CLASSES)):
            currentWeights[c] = self.updateWeightsPerClasStep(currentWeights[c], trainingSamples, self.CLASSES[c], self.LEARNING_RATE * (self.SHRINKAGE ** step))


        #check the current error and compute accuracy, then do a debug output to see the progress
        currentGeneralError, currentConfusionMatrix = self.helper.calcTotalError(self, trainingSamples, currentWeights)
        currentAccuracy = 1- currentGeneralError
        print("Pr obal Weight: " + str(step) + " Right: " + str(1-currentGeneralError) + self.helper.strRuntime(self.start))
        self.accuracy.append(currentAccuracy) #save accuracy for later result printing

        if (testSet != None):
            errorBeforeTest, confusionMatrixTest = self.helper.calcTotalError(self, testSet, currentWeights)
            accuracyStepTest = 1-errorBeforeTest

            self.accuracyTestSet.append(accuracyStepTest)

            print("\tAcc: " + str("%.4f" % accuracyStepTest))

        #check if we need to store the new accuracy as the new best one
        if(currentAccuracy > self.accuracy[self.maxAccuracyIndex]):
            self.maxAccuracyIndex = step
            self.maxWeights = copy.deepcopy(currentWeights)

        #Check if we print the confusion matrix
        if(step % 10 == 0):
            self.helper.writeConfusionMatrixToFile(currentConfusionMatrix, self.CLASSES, self.confusionFilenameTemplate + "_step_" + str(step) + "_confusion.txt")

        self.endTime = time.time()

        return currentWeights

    #Will optimize the weights for one class only. Thereby this will only do one step of gradient decent.
    #CurrentWeightsPerClass is the vector contining the weights for this class logistic regression. Training Samples is a list of training samples. Current Class is nominal (string) class value.
    def updateWeightsPerClasStep(self, currentWeightsPerClass, trainingSamples, currentClass, shrinkedLearningRate):
        newWeights = currentWeightsPerClass
        deltaW = np.zeros(len(currentWeightsPerClass))

        #do for all samples,
        for sample in trainingSamples:
            sampleInput = sample[0]
            sampleTarget = sample[1]
            #The target number is either -1 (this class is not the real one) or 1.
            target = -1
            if sampleTarget == currentClass:
                target = 1

            phi = self.helper.getPhi(sampleInput, self.BASIS_FUNCTION)
            #See paper for more beautiful formulas.
            wTimesPhi = np.dot(np.transpose(currentWeightsPerClass), phi) * target

            if wTimesPhi < 1:
                deltaW += np.multiply(-1 * target, phi)

        #update w with learning rate of its gradient.
        newWeights = newWeights - deltaW * shrinkedLearningRate

        return newWeights

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
