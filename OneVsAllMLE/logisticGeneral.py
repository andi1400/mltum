import csv
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import datetime as date
import signal
import sys
import getopt
from MLEOneVsAll import mleonevsall
from softZeroOneLoss import softzeroone
from helper import *
from hingeLoss import hinge

class logisticregression():
    NOTES = "using approximated sigmoid, oneVSall, noBasis"
    TEST_AND_PLOT_FITNESS = False
    STORERESULTS = True

    #The possible classes, we will use the index to identify the classes in the classifier
    CLASSES = ["sitting", "walking", "standing", "standingup", "sittingdown"]

    learnMethod = None
    helper = None


    csvRunFilename = "../output/experiments.csv"
    csvWeightsFilename = "../output/weights/run_"


    def __init__(self, learnMethod, helper):
        self.helper = helper
        self.learnMethod = learnMethod(self.CLASSES)

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

    #write a list of lists to the central csv file. One line per inner list.
    def writeToCSV(self,filename):
        with open(filename, "a") as file:
            csvwriter = csv.writer(file, delimiter=";", quotechar='"')

            #write all that has been given
            csvwriter.writerow([date.datetime.fromtimestamp(self.learnMethod.getStartTime())] + self.learnMethod.getParameterNameList())
            csvwriter.writerow([str(self.learnMethod)] + self.learnMethod.getParameterList())
            csvwriter.writerow([" "] + self.learnMethod.getAccuracy())

            #a new line
            csvwriter.writerow([])


    def printRunInformation(self):
        plt.clf()
        plt.plot(self.learnMethod.getAccuracy())
        plt.axis([0,len(self.learnMethod.getAccuracy()),0,1])
        plt.draw()

        if self.STORERESULTS:
            self.writeToCSV(self.csvRunFilename)
            self.helper.writeWeights(self.csvWeightsFilename + str(self.learnMethod.getStartTime()) + str(self.learnMethod) +".csv", self.CLASSES, self.learnMethod.getWeights(), True)

            #create a plot
            plt.savefig("../output/plots/run_" + str(date.datetime.fromtimestamp(self.learnMethod.getStartTime())) + ".png")
            plt.savefig("../output/plots/run_" + str(date.datetime.fromtimestamp(self.learnMethod.getStartTime())) + ".pdf")

        currentError, confusionMatrix = helper.calcTotalError(self.learnMethod, originalData, self.learnMethod.getWeights())
        print("Best Error on training: " + str(currentError))
        print("Best Accuracy on training: " + str(1-currentError))

        print("____________________________")
        print(self.learnMethod.getWeights())
        print("____________________________")

        print(self.helper.getConfusionMatrixAsString(confusionMatrix, self.CLASSES))
        self.helper.writeConfusionMatrixToFile(confusionMatrix, self.CLASSES, self.learnMethod.confusionFilenameTemplate + "_FINAL_confusion.txt")

        plt.show(block=True)

    def train(self, trainingSamples, startWeights):
        self.learnMethod.learn(startWeights, trainingSamples)

################################
#general control flow section
################################
helper = helper()
#learnMethod = mleonevsall
#learnMethod = softzeroone
learnMethod = hinge

logisticregressionHelper = logisticregression(learnMethod, helper)

print("Running " + str(learnMethod))
print(logisticregressionHelper.learnMethod.getParameterNameList())
print(logisticregressionHelper.learnMethod.getParameterList())

#create start weights or read them
startWeights = []
for i in range(len(logisticregressionHelper.CLASSES)):
    dummyWeight = np.zeros(17)
    startWeights.append(dummyWeight)


#read cmd line arguments
#try:
print("About to checking arguments...")
for i in range(len(sys.argv)):
    if sys.argv[i] == "-h" or sys.argv[i] == "--help":
        print("Here should be your help.")
        sys.exit()
    elif sys.argv[i] == "-w" or sys.argv[i] == "--weights":
        print("received weights - processing them...")
        weightFile = sys.argv[i+1]
        i += 1
        startWeights = helper.readWeights(weightFile, logisticregressionHelper.CLASSES)

#read the data
originalData = logisticregressionHelper.readData("../data/dataset-complete_90PercentTrainingSet_mini10Percent_standardized.arff")
#originalData = logisticregressionHelper.readData("../data/dataset-complete_90PercentTrainingSet_normalized.arff")
print("Test reading: " + str(originalData[0]))

#catch STRG+C to prevent loss of output.
def signal_handler(signal, frame):
    logisticregressionHelper.printRunInformation()
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


#Train the model and print the run information
logisticregressionHelper.train(originalData, startWeights)
logisticregressionHelper.printRunInformation()


