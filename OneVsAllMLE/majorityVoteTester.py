from helper import helper
import numpy as np
import time


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
    maxAccuracyIndex = 0
    maxWeights = None

    debugFolderName = None
    weightsFilenameTemplate = None
    confusionFilenameTemplate = None

    def __init__(self, CLASSES):
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

    def learn(self, startWeights, trainingSamples):
        self.start = time.time()

        #set the debug filenames and create folders
        self.setFilenames()

        unitWeights = []
        for i in range(len(self.classifierAndIndividualWeights)):
            None
            #unitWeights.append(1)
        #TODO TEST
        unitWeights.append(0.6629)
        unitWeights.append(0.7241)
        self.endTime = time.time()

        return unitWeights

    def classifySample(self, sample, classifierMetaWeights):
        if(classifierMetaWeights is None):
            classifierMetaWeights = self.learn(None, None)

        currentResult = np.zeros(len(self.CLASSES))

        for classifierIdx in range(len(self.classifierAndIndividualWeights)):
            classifierWeights = self.classifierAndIndividualWeights[classifierIdx][1]
            classifier = self.classifierAndIndividualWeights[classifierIdx][0]

            predictedClass, confidenceOfPredicted, classPercentages = classifier.classifySample(sample, classifierWeights)

            for classIdx in range(len(self.CLASSES)):
                currentResult[classIdx] += classPercentages[classIdx] * classifierMetaWeights[classifierIdx]



        sumX = sum(currentResult)
        classPercentagesNormalized = [x/sumX for x in currentResult]
        confidenceOfPredicted = max(classPercentagesNormalized)

        predictedClass = classPercentagesNormalized.index(confidenceOfPredicted)
        predictedClass = self.CLASSES[predictedClass]

        return predictedClass, confidenceOfPredicted, classPercentagesNormalized

    def getAccuracy(self):
        return self.accuracy

    def getWeights(self):
        return None

    def getParameterNameList(self):
        return self.parameterNames

    def getParameterList(self):
        return self.parameters

    def getStartTime(self):
        return self.start