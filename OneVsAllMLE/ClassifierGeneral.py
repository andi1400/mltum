import csv
import matplotlib.pyplot as plt
import datetime as date
import signal
import sys
from MLEOneVsAll import mleonevsall
from softZeroOneLoss import softzeroone
from helper import *
from hingeLoss import hinge
from majorityVoteTester import majorityvote
from weightedClassifiers import weightedclassifiers
from neuralnetwork import neuralnetwork
from neuralnetworkDropout import neuralnetworkDropout


"""
This class aggregates all general features necessary for classification.
 This includes the control flow, output, parameter setting, classifier initialization, plotting and bookkeeping.
"""
class classifierGeneral():

    #If true, results will be stored in a file for later usage. Turn it off for just normal testing.
    STORERESULTS = True

    #The possible classes, we will use the index to identify the classes in the classifier
    CLASSES = None

    #The classifier method that is used with this instance.
    classifierMethod = None

    #A helper instance.
    helper = None

    #The filename for the experiments.csv, which is the file where all experiment data will be appended.
    csvRunFilename = "../output/experiments.csv"

    #And the general style for output files.
    csvWeightsFilename = "../output/weights/run_"


    # Initializes the ClassifierGeneral. It requires a classifier method instance, and a helper instance.
    def __init__(self, classifierMethod, helper):
        self.helper = helper
        self.classifierMethod = classifierMethod

    #Writes the experimental results of this run to the main file.
    def writeToRunFile(self):
        with open(self.csvRunFilename, "a") as file:
            csvwriter = csv.writer(file, delimiter=";", quotechar='"')

            #write all that has been given
            csvwriter.writerow([date.datetime.fromtimestamp(self.classifierMethod.getStartTime())] + self.classifierMethod.getParameterNameList())
            csvwriter.writerow([str(self.classifierMethod)] + self.classifierMethod.getParameterList())
            csvwriter.writerow([" "] + self.classifierMethod.getAccuracy())

            #a new line
            csvwriter.writerow([])

    # Prints the run information to the console. Plots the accuracy over time. If storeresults is true, it will store the results, too.
    def printRunInformation(self):
        #Initialize and draw plot.
        plt.clf()
        if (self.classifierMethod.accuracyTestSet != None):
            plt.plot(self.classifierMethod.accuracyTestSet)
        plt.plot(self.classifierMethod.getAccuracy())
        plt.axis([0,len(self.classifierMethod.getAccuracy()),0,1])
        plt.draw()

        # Print errors on test and data set.
        print("Evaluating Model - Please wait...")
        currentError, confusionMatrix = self.helper.calcTotalError(self.classifierMethod, testData, self.classifierMethod.getWeights())
        currentErrorTest, confusionMatrixTest = self.helper.calcTotalError(self.classifierMethod, testData, self.classifierMethod.getWeights())
        print("Last Error on training: " + str(currentError))
        print("Last Accuracy on training: " + str(1-currentError))
        print("Last Error on test: " + str(currentErrorTest))
        print("Last Accuracy on test: " + str(1-currentErrorTest))

        # Print weights.
        print("____________________________")
        print(self.classifierMethod.getWeights())
        print("____________________________")

        #Print the confusion matrix.
        print(self.helper.getConfusionMatrixAsString(confusionMatrix, self.CLASSES))

        #Store the results.
        if self.STORERESULTS:
            self.writeToRunFile()
            self.helper.writeWeightsDebug(self.csvWeightsFilename + str(self.classifierMethod.getStartTime()) + str(self.classifierMethod) +".csv", self.classifierMethod.getWeights())
            #store a plot
            plt.savefig("../output/plots/run_" + str(self.classifierMethod.__class__.__name__) + "_" + str(date.datetime.fromtimestamp(self.classifierMethod.getStartTime())) + ".png")
            plt.savefig("../output/plots/run_" + str(self.classifierMethod.__class__.__name__) + "_" + str(date.datetime.fromtimestamp(self.classifierMethod.getStartTime())) + ".pdf")
            self.helper.writeConfusionMatrixToFile(confusionMatrix, self.CLASSES, self.classifierMethod.confusionFilenameTemplate + "_FINAL_confusionTraining.txt")
            self.helper.writeConfusionMatrixToFile(confusionMatrixTest, self.CLASSES, self.classifierMethod.confusionFilenameTemplate + "_FINAL_confusionTest.txt")

            self.helper.writeAccuracies("../output/accuracies/run_" + str(self.classifierMethod.__class__.__name__) + "_" + str(date.datetime.fromtimestamp(self.classifierMethod.getStartTime())) + ".csv", self.classifierMethod.getAccuracy(), self.classifierMethod.accuracyTestSet)

        #And show the plot, blocking all inputs.
        plt.show(block=True)

    #Trains the classifier on the training samples. The test data will be used to plot the accuracy for every step; it is not required to learn.
    def train(self, trainingSamples, startWeights, testData=None):
        if startWeights is None:
            self.classifierMethod.learn(trainingSamples, None, testData)
        else:
            self.classifierMethod.learn(trainingSamples, startWeights, testData)

################################
#general control flow section
################################

#Using this, we can interrupt the program without loss of data - once it is ended, it will automatically call printRunInformation.
#Except if it already ended, in that case, it will just exit.
def signal_handler(signal, frame):
    global terminate
    if not terminate:
        terminate = True
        classifierGeneralInstance.printRunInformation()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


#The definitions for methods and classes.

#This defines the classes.
CLASSES = ["sitting", "walking", "standing", "standingup", "sittingdown"]

#This defines the available classifiers and the console parameter it can be started with.
CLASSIFIERS = {'MLE': mleonevsall, 'SOFTZEROONE': softzeroone, 'HINGE': hinge, 'MAV': majorityvote, 'WAVG': weightedclassifiers, 'NN': neuralnetwork, 'NND': neuralnetworkDropout}

#This defines the parameters they will be given.
PARAMETERS = {'MLE': [6e-5, 0.991], 'SOFTZEROONE': [0.0001, 0.99993, 2.5, 1e-7], 'HINGE': [8e-5, 0.9995], 'MAV': None, 'WAVG': None, 'NN': [1e-2, 1, 4, [16, 50, 50, 5]], 'NND': [1e-2, 1, 4, [16, 50, 50, 5]]}

MAXSTEPS = 100000
MAXNONCHANGINGSTEPS = 1000
helper = helper()

classifierGeneralInstance = None
noLearn = False
terminate = False
startWeights = None
trainingDataFilename = None
testDataFilename = None

#read cmd line arguments
#try:
print("About to checking arguments...")

for i in range(len(sys.argv)):
    if sys.argv[i] == "-h" or sys.argv[i] == "--help":
        print("ClassifierGeneral help:")
        print("\t Usage: python ClassifierGeneral -c [-w, -nl, -ns, -lD, -tD]")
        print("\t -c or --classifier: Sets the classifier to be used. Possible arguments:")
        print("\t\t" + str(CLASSIFIERS))
        print("\t -w or --weights: Sets the weight data to use for initial weights. Format as printed out.")
        print("\t -nl or --nolearn: Disables learning; just classifies.")
        print("\t -ns or --nostore: Disables the output to file. Use this for testing.")
        print("\t -ld or --learningdata: Sets the file for the training data. If none is given, a standard set will be used.")
        print("\t -td or --testdata: Sets the file for the test data. If none is given, a standard set will be used.")
        print("\t -h or --help: Prints this help. You knew this, right?")
        sys.exit()
    elif sys.argv[i] == "-c" or sys.argv[i] == "--classifier":
        #Defines the classifier to be used.
        classifierMethod = CLASSIFIERS[sys.argv[i+1]](CLASSES, MAXSTEPS, MAXNONCHANGINGSTEPS, PARAMETERS[sys.argv[i+1]])

        if sys.argv[i+1] == 'MAV' or sys.argv[i+1] == 'WAVG':
            classifierNames = sys.argv[i+2].split(',')
            #now instantiate all the classifier instances
            classifiersWithWeightsForMav = []
            for name in classifierNames:
                instance = CLASSIFIERS[name](CLASSES, MAXSTEPS, MAXNONCHANGINGSTEPS, PARAMETERS[name])
                weights = helper.readWeights("MAVWeights/" + name + ".weights", CLASSES)
                classifierWithWeights = [instance, weights]
                classifiersWithWeightsForMav.append(classifierWithWeights)#

            classifierMethod.setClassifiers(classifiersWithWeightsForMav)

        classifierGeneralInstance = classifierGeneral(classifierMethod, helper)
        print(classifierGeneralInstance)

    elif sys.argv[i] == "-w" or sys.argv[i] == "--weights":
        #This defines the weights.
        print("received weights data - processing them...")
        weightFile = sys.argv[i+1]
        i += 1
        startWeights = helper.readWeights(weightFile, CLASSES)
        classifierGeneralInstance.classifierMethod.maxWeights = startWeights

    elif sys.argv[i] == "-nl" or sys.argv[i] == "--nolearn":
        print("LEARNING DISABLED - ONLY CLASSIFICATION")
        noLearn = True
        classifierGeneralInstance.STORERESULTS = False

    elif sys.argv[i] == "-ns" or sys.argv[i] == "--nostore":
        print("Run information storage off.")
        classifierGeneralInstance.STORERESULTS = False

    elif sys.argv[i] == "-ld" or sys.argv[i] == "--learningdata":
        trainingDataFilename = sys.argv[i+1]
        i += 1

    elif sys.argv[i] == "-td" or sys.argv[i] == "--testdata":
        testDataFilename = sys.argv[i+1]
        i += 1




classifierGeneral.CLASSES = CLASSES



print("Running " + str(classifierMethod))

#read the data. If no data file has been set, take a standard one.
if trainingDataFilename == None:
    trainingDataFilename = "../data/TRAINING.arff"

if testDataFilename == None:
    testDataFilename = "../data/TEST.arff"


trainingData = classifierGeneralInstance.helper.readData(trainingDataFilename)
testData = classifierGeneralInstance.helper.readData(testDataFilename)
print("Using dataset: " + trainingDataFilename)
print("Test reading: " + str(trainingData[0]))

#Train the model and print the run information
if(not noLearn):
   print("\n\n------------------------------")
   print("Entering learning mode...")
   print("------------------------------\n")
   print(classifierGeneralInstance.classifierMethod.getParameterNameList())
   print(classifierGeneralInstance.classifierMethod.getParameterList())

   classifierGeneralInstance.train(trainingData, startWeights, testData)

classifierGeneralInstance.printRunInformation()


