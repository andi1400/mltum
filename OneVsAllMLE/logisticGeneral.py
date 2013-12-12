import csv
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import datetime as date
import signal
import sys
from MLEOneVsAll import mleonevsall
from softZeroOneLoss import softzeroone

class logisticregression():
    NOTES = "using approximated sigmoid, oneVSall, noBasis"
    TEST_AND_PLOT_FITNESS = False
    STORERESULTS = True

    #The possible classes, we will use the index to identify the classes in the classifier
    CLASSES = ["sitting", "walking", "standing", "standingup", "sittingdown"]

    LEARNING_RATE = 1e-5
    SHRINKAGE = 0.95

    MAX_STEPS = 100000000
    UPDATE_THRESHOLD = 1e-10
    MAX_NONCHANGING_STEPS = 100000000

    beta = 2
    gradientstepsize = 0.1
    regularizer = 0 #lambda

    start = time.time()
    learnMethod = None

    def __init__(self, learnMethod):
        #self.learnMethod = learnMethod(self.CLASSES, self.LEARNING_RATE, self.SHRINKAGE, self.MAX_STEPS, self.MAX_NONCHANGING_STEPS, self.UPDATE_THRESHOLD, self.start, self)
        self.learnMethod = learnMethod(self.CLASSES, self.LEARNING_RATE, self.SHRINKAGE, self.MAX_STEPS, self.MAX_NONCHANGING_STEPS, self.UPDATE_THRESHOLD, self.start, self, self.beta, self.regularizer, self.gradientstepsize)

    #read the data from an ARFF file.
    def readData(self, filename):
        data = []
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=",", quotechar='"')
            dataReached = False
            for row in reader:
                if(len(row) > 0 and row[0] == "@data"):
                    dataReached = True
                    continue

                if(dataReached):
                    dataRow = [[]]
                    for xIndex in range(2, len(row)-1):
                        dataRow[0].append(float(row[xIndex]))
                    dataRow.append(row[-1])
                    data.append(dataRow)
        return data

    def writeToCSV(self,filename):
        runtimeSeconds = time.time() - self.start
        metaValues = [date.datetime.fromtimestamp(self.start), runtimeSeconds, learnMethod.fitness[learnMethod.maxFitnessIndex], self.LEARNING_RATE, self.SHRINKAGE, self.MAX_STEPS, self.MAX_NONCHANGING_STEPS, self.NOTES, self.beta, self.regularizer, str(self.learnMethod), self.gradientstepsize]

        with open(filename, "a") as file:
            csvwriter = csv.writer(file, delimiter=";", quotechar='"')
            csvwriter.writerow(metaValues)
            csvwriter.writerow([""]+learnMethod.fitness)
            csvwriter.writerow([])


    def writeWeights(self, filenameTemplate):
        filename = filenameTemplate + str(date.datetime.fromtimestamp(self.start)) + ".csv"
        with open(filename, "a") as file:
            csvwriter = csv.writer(file, delimiter=";", quotechar='"')

            for i in range(len(self.CLASSES)):
                csvwriter.writerow([self.CLASSES[i]] + learnMethod.maxWeights[i].tolist())

    def calcTotalError(self, weights, trainingSamples):
        curRight = 0
        for i in range(len(trainingSamples)):

            if self.classifySample(trainingSamples[i][0], weights)[0] == trainingSamples[i][1]:
                curRight += 1


        #print("Correctly Classified Samples: " + str(curRight))

        return 1-float(curRight)/len(trainingSamples)

    #will return three values, the first is the class, the second the confidence for the highest class, the third an array of normalized confidences for each class
    #ClassWeights containing the weight vector per class as columns
    def classifySampleOld(self, x, ClassWeights):
        classPercentages = np.zeros(len(self.CLASSES))
        for i in range(len(self.CLASSES)):
            currentWeightVector = ClassWeights[i]
            currentFeatureVector = self.getPhi(x)

            wTimesPhi = np.dot(np.transpose(currentWeightVector), currentFeatureVector)
            regressionResult = np.exp(wTimesPhi)
            classPercentages[i] = regressionResult

        sumX = sum(classPercentages)
        classPercentagesNormalized = [x/sumX for x in classPercentages]
        confidenceOfPredicted = max(classPercentagesNormalized)
        predictedClass = self.CLASSES[classPercentagesNormalized.index(confidenceOfPredicted)]

        return predictedClass, confidenceOfPredicted, classPercentages


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
        currentFeatureVector = self.getPhi(x)

        wTimesPhi = np.dot(np.transpose(ClassWeight), currentFeatureVector)
       # print(wTimesPhi)
        regressionResult = self.sigmoid(wTimesPhi)

        if(regressionResult >= 0.5):
            return 1, regressionResult

        return 0, regressionResult


    #returns a vector containing as first element the bias b and the others are the features for sample x derived by applying
    #the basis function from getBasisOutput on each
    def getPhi(self, x):
        returnVector = np.zeros(len(x)+1)

        for i in range(len(x)+1):
            returnVector[i] = self.getBasisOutput(x, i)

        return returnVector

    def getBasisOutput(self, x, i):
        if(i == 0):
            return 1
        return x[i-1]

    def sigmoid(self, x):
        return 0.5 * (x/(1+abs(x)) + 1)
        #return 1 / (1 - np.exp(-x))

    ###############
    #Helpers
    ###############
    def runtime(self):
        return " runtime(s): " + str(time.time() - self.start)

    def signal_handler(self, signal, frame):
        plt.clf()
        plt.plot(learnMethod.fitness)
        plt.axis([0,len(learnMethod.fitness),0,1])
        plt.draw()

        if self.STORERESULTS:
            self.writeToCSV("../output/experiments.csv")
            self.writeWeights("../output/weights/run_")

            #create a plot
            plt.savefig("../output/plots/run_" + str(date.datetime.fromtimestamp(self.start)) + ".png")
            plt.savefig("../output/plots/run_" + str(date.datetime.fromtimestamp(self.start)) + ".pdf")

        plt.show(block=True)
        sys.exit(0)

    def train(self, trainingSamples):
        curWeights = []
        for i in range(len(self.CLASSES)):
            dummyWeight = np.zeros(17)
            curWeights.append(dummyWeight)

        for i in range(self.MAX_STEPS):
            curWeights = self.learnMethod.optimizeAllWeights(curWeights, trainingSamples, i)

    ################################
    #general control flow section
    ################################

#learnMethod = mleonevsall
learnMethod = softzeroone
logisticregressionHelper = logisticregression(learnMethod)


originalData = logisticregressionHelper.readData("../data/dataset-complete_90PercentTrainingSet_mini10Percent.arff")
signal.signal(signal.SIGINT, logisticregressionHelper.signal_handler)


print("Test reading: " + str(originalData[0]))

logisticregressionHelper.train(originalData)
logisticregressionHelper.signal_handler(None, None)

currentError = logisticregressionHelper.calcTotalError(logisticregressionHelper.learnMethod.maxWeights,originalData)
print("Best Error on training: " + str(currentError) + logisticregressionHelper.runtime())
print("Best Accuracy on training: " + str(1-currentError) + logisticregressionHelper.runtime())

print("____________________________")
print(logisticregressionHelper.learnMethod.maxWeights)
print("____________________________")


input("stopped")

