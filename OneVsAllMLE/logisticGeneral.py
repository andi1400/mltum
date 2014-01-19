# cython: profile=True

import csv
import numpy as np
import math
import time
import cProfile
import re
import matplotlib.pyplot as plt
import datetime as date
import signal
import sys
import getopt
from MLEOneVsAll import mleonevsall
from softZeroOneLoss import softzeroone
from helper import *
from hingeLoss import hinge
from majorityVoteTester import majorityvote
from weightedClassifiers import weightedclassifiers
from neuralnetwork import neuralnetwork
from neuralnetworkNew import neuralnetworkNew
#from neuralnetworkCCode import neuralnetwork

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
        self.learnMethod = learnMethod

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
        if (self.learnMethod.accuracyTestSet != None):
            plt.plot(self.learnMethod.accuracyTestSet)
        plt.plot(self.learnMethod.getAccuracy())
        plt.axis([0,len(self.learnMethod.getAccuracy()),0,1])
        plt.draw()


        print("Evaluating Model - Please wait...")
        currentError, confusionMatrix = self.helper.calcTotalError(self.learnMethod, originalData, self.learnMethod.getWeights())
        currentErrorTest, confusionMatrixTest = self.helper.calcTotalError(self.learnMethod, testData, self.learnMethod.getWeights())
        print("Last Error on training: " + str(currentError))
        print("Last Accuracy on training: " + str(1-currentError))
        print("Last Error on test: " + str(currentErrorTest))
        print("Last Accuracy on test: " + str(1-currentErrorTest))

        print("____________________________")
        print(self.learnMethod.getWeights())
        print("____________________________")

        print(self.helper.getConfusionMatrixAsString(confusionMatrix, self.CLASSES))

        if self.STORERESULTS:
            self.writeToCSV(self.csvRunFilename)
            self.helper.writeWeightsDebug(self.csvWeightsFilename + str(self.learnMethod.getStartTime()) + str(self.learnMethod) +".csv", self.learnMethod.getWeights())
            #create a plot

            plt.savefig("../output/plots/run_" + str(self.learnMethod.__class__.__name__) + "_" + str(date.datetime.fromtimestamp(self.learnMethod.getStartTime())) + ".png")
            plt.savefig("../output/plots/run_" + str(self.learnMethod.__class__.__name__) + "_" + str(date.datetime.fromtimestamp(self.learnMethod.getStartTime())) + ".pdf")
            self.helper.writeConfusionMatrixToFile(confusionMatrix, self.CLASSES, self.learnMethod.confusionFilenameTemplate + "_FINAL_confusionTraining.txt")
            self.helper.writeConfusionMatrixToFile(confusionMatrixTest, self.CLASSES, self.learnMethod.confusionFilenameTemplate + "_FINAL_confusionTest.txt")

            self.helper.writeAccuracies("../output/accuracies/run_" + str(self.learnMethod.__class__.__name__) + "_" + str(date.datetime.fromtimestamp(self.learnMethod.getStartTime())) + ".csv", self.learnMethod.getAccuracy(), self.learnMethod.accuracyTestSet)

        plt.show(block=True)

    def train(self, trainingSamples, startWeights, testData=None):
        if startWeights is None:
            self.learnMethod.learn(trainingSamples, None, testData)
        else:
            self.learnMethod.learn(trainingSamples, startWeights, testData)

################################
#general control flow section
################################

def signal_handler(signal, frame):
    global terminate
    if not terminate:
        terminate = True
        logisticregressionHelper.printRunInformation()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

MAXSTEPS = 100000
MAXNONCHANGINGSTEPS = 1000
helperInstance = helper()
terminate = False
logisticregressionHelper = None
originalData = None

#run()
CLASSES = ["sitting", "walking", "standing", "standingup", "sittingdown"]
CLASSIFIERS = {'MLE': mleonevsall, 'SOFTZEROONE': softzeroone, 'HINGE': hinge, 'MAV': majorityvote, 'WAVG': weightedclassifiers, 'NN': neuralnetwork, 'NNNew': neuralnetworkNew}
PARAMETERS = {'MLE': [6e-5, 0.991], 'SOFTZEROONE': [0.0001, 0.99993, 2.5, 1e-7], 'HINGE': [8e-5, 0.9995], 'MAV': None, 'WAVG': None, 'NN': [1e-2, 1, 5, [16, 100, 100, 100, 5]], 'NNNew': [1e-3, 1, 4, [16, 1000, 1000, 5]]}
#
MAXSTEPS = 100000
MAXNONCHANGINGSTEPS = 1000
helper = helper()
#learnMethod = mleonevsall
#learnMethod = softzeroone
#learnMethod = hinge(CLASSES)

logisticregressionHelper = None
noLearn = False
terminate = False

#create start weights or read them
startWeights = None

#read cmd line arguments
#try:
print("About to checking arguments...")

for i in range(len(sys.argv)):
   if sys.argv[i] == "-h" or sys.argv[i] == "--help":
       print("Here should be your help.")
       sys.exit()
   elif sys.argv[i] == "-c" or sys.argv[i] == "--classifier":
       learnMethod = CLASSIFIERS[sys.argv[i+1]](CLASSES, MAXSTEPS, MAXNONCHANGINGSTEPS, PARAMETERS[sys.argv[i+1]])

       if sys.argv[i+1] == 'MAV' or sys.argv[i+1] == 'WAVG':
           classifierNames = sys.argv[i+2].split(',')
            #now instantiate all the classifier instances
           classifiersWithWeightsForMav = []
           for name in classifierNames:
               instance = CLASSIFIERS[name](CLASSES, MAXSTEPS, MAXNONCHANGINGSTEPS, PARAMETERS[name])
               weights = helper.readWeights("MAVWeights/" + name + ".weights", CLASSES)
               classifierWithWeights = [instance, weights]
               classifiersWithWeightsForMav.append(classifierWithWeights)#

           learnMethod.setClassifiers(classifiersWithWeightsForMav)

       logisticregressionHelper = logisticregression(learnMethod, helper)
       print(logisticregressionHelper)

   elif sys.argv[i] == "-w" or sys.argv[i] == "--weights":
       print("received weights - processing them...")
       weightFile = sys.argv[i+1]
       i += 1
       startWeights = helper.readWeights(weightFile, CLASSES)
       logisticregressionHelper.learnMethod.maxWeights = startWeights

   elif sys.argv[i] == "-nl" or sys.argv[i] == "--nolearn":
       print("LEARNING DISABLED - ONLY CLASSIFICATION")
       noLearn = True
       logisticregressionHelper.STORERESULTS = False



print("Running " + str(learnMethod))

#read the data
#dsfilename = "../data/dataset-complete_90PercentTrainingSet_mini10Percent_standardized.arff"
#dsfilename = "../data/testDataSetTraining_5percent_standardized.arff"
#dsfilename = "../data/dataset-complete_90PercentTrainingSet_mini10Percent_normalized_only149.arff"
#dsfilename = "../data/dataset-complete_90PercentTrainingSet_normalized.arff"
#dsfilename = "../data/dataset-complete_90PercentTrainingSet_standardized.arff"
dsfilename = "../data/TRAINING.arff"
#dsfilename = "../data/testDataSetTraining_5percent_standardized.arff"

#testFilename = "../data/dataset-complete_10PercentTestSet_standardized.arff"
testFilename = "../data/TEST.arff"
#testFilename = "../data/testDataSetTest_5percent_standardized.arff"

originalData = logisticregressionHelper.helper.readData(dsfilename)
testData = logisticregressionHelper.helper.readData(testFilename)
print("Using dataset: " + dsfilename)
print("Test reading: " + str(originalData[0]))

#catch STRG+C to prevent loss of output.
def signal_handler(signal, frame):
   global terminate
   if not terminate:
       terminate = True
       logisticregressionHelper.printRunInformation()
   sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


#Train the model and print the run information
if(not noLearn):
   print("\n\n------------------------------")
   print("Entering learning mode...")
   print("------------------------------\n")
   print(logisticregressionHelper.learnMethod.getParameterNameList())
   print(logisticregressionHelper.learnMethod.getParameterList())

   logisticregressionHelper.train(originalData, startWeights, testData)

logisticregressionHelper.printRunInformation()


