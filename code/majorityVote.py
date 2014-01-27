from helper import helper
import numpy as np
import time


#Implements a majority vote classifier.
class majorityvote:
    CLASSES = None

    #holds pairs of classifier objects and their weight matrices
    classifierAndIndividualWeights = []

    helper = None

    parameterNames = []
    parameters = []

    start = None
    endTime = None

    accuracy = []
    accuracyTestSet = None
    maxAccuracyIndex = 0
    maxWeights = None

    debugFolderName = None
    weightsFilenameTemplate = None
    confusionFilenameTemplate = None
    #Initializes it. Only the classes are important.
    def __init__(self, CLASSES, maxSteps, maxNonChangingSteps, parameters):

        self.helper = helper()
        self.CLASSES = CLASSES

    def setFilenames(self):
        self.debugFolderName = "../output/weights/debug/" + str(self.start) + "_" + str(self.__class__.__name__) + "/"
        self.weightsFilenameTemplate = self.debugFolderName + str(self.start)
        self.confusionFilenameTemplate = self.debugFolderName + str(self.start)


    def setClassifiers(self, classifiersAndWeights):
        self.classifierAndIndividualWeights = classifiersAndWeights

        for i in range(len(self.classifierAndIndividualWeights)):
            self.parameterNames.append("Classifier" + str(i))
            self.parameters.append(self.classifierAndIndividualWeights[i][0].__class__.__name__)

    def setUniformWeights(self):
        unitWeights = []
        for i in range(len(self.classifierAndIndividualWeights)):
            unitWeights.append(1)

        return unitWeights

    #dummy only!!
    def learn(self, startWeights, trainingSamples):
       self.maxWeights = self.setUniformWeights()

    def classifySample(self, sample, classifierMetaWeights):
        if(classifierMetaWeights is None):
            #set the debug filenames and create folders
            self.setFilenames()
            classifierMetaWeights = self.setUniformWeights()

        resultPredictions = np.zeros(len(self.CLASSES))
        individualResults = []
        individualAccuracy = []

        for classifierIdx in range(len(self.classifierAndIndividualWeights)):
            classifierWeights = self.classifierAndIndividualWeights[classifierIdx][1]
            classifier = self.classifierAndIndividualWeights[classifierIdx][0]

            predictedClass, confidenceOfPredicted, classPercentages = classifier.classifySample(sample, classifierWeights)

            resultPredictions[self.CLASSES.index(predictedClass)] += 1
            individualResults.append(predictedClass)
            individualAccuracy.append(confidenceOfPredicted)

        prediction = None

        maxValue = max(resultPredictions)
        maxPredictors = []
        for i in range(len(resultPredictions)):
            if resultPredictions[i] == maxValue:
                maxPredictors.append(i)

        #decide what is the majority class
        if(len(maxPredictors) == 1):
            prediction = self.CLASSES[maxPredictors[0]]
        else:
            tieClassifierConfidences = []
            for i in range(len(maxPredictors)):
                tieClassifierConfidences.append(individualAccuracy[i])
            prediction = individualResults[tieClassifierConfidences.index((max(tieClassifierConfidences)))]

        classPercentagesNormalized = [x/len(self.classifierAndIndividualWeights) for x in resultPredictions]
        confidenceOfPredicted = max(classPercentagesNormalized)
        return prediction, confidenceOfPredicted, classPercentagesNormalized

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