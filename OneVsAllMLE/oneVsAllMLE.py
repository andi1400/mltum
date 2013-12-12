import csv
import numpy as np
import math
import time

#The possible classes, we will use the index to identify the classes in the classifier
CLASSES = ["sitting", "walking", "standing", "standingup", "sittingdown"]
LEARNING_RATE = 0.001
MAX_STEPS = 50
UPDATE_THRESHOLD = 1e-10
start = time.time()

#read the data from an ARFF file.
def readData(filename):
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

def calcTotalError(weights, trainingSamples):
    curRight = 0
    for i in range(len(trainingSamples)):

        #if classifySample(trainingSamples[i][0], weights)[0] != "sittingdown":
        #    print("Prediction: " + str(classifySample(trainingSamples[i][0], weights)))
        #    print("Label: " + str(trainingSamples[i][1]))
        if classifySample(trainingSamples[i][0], weights)[0] == trainingSamples[i][1]:
            curRight += 1


    #print("Correctly Classified Samples: " + str(curRight))

    return 1-float(curRight)/len(trainingSamples)

#will return three values, the first is the class, the second the confidence for the highest class, the third an array of normalized confidences for each class
#ClassWeights containing the weight vector per class as columns
def classifySampleOld(x, ClassWeights):
    classPercentages = np.zeros(len(CLASSES))
    for i in range(len(CLASSES)):
        currentWeightVector = ClassWeights[i]
        currentFeatureVector = getPhi(x)

        wTimesPhi = np.dot(np.transpose(currentWeightVector), currentFeatureVector)
        regressionResult = np.exp(wTimesPhi)
        classPercentages[i] = regressionResult

    sumX = sum(classPercentages)
    classPercentagesNormalized = [x/sumX for x in classPercentages]
    confidenceOfPredicted = max(classPercentagesNormalized)
    predictedClass = CLASSES[classPercentagesNormalized.index(confidenceOfPredicted)]

    return predictedClass, confidenceOfPredicted, classPercentages


def classifySample(x, ClassWeights):
    classPercentages = np.zeros(len(CLASSES))
    for i in range(len(CLASSES)):
        currentWeightVector = ClassWeights[i]

        classPercentages[i] = classifySampleSingleClass(x, currentWeightVector)[1]

    sumX = sum(classPercentages)
    classPercentagesNormalized = [x/sumX for x in classPercentages]
    confidenceOfPredicted = max(classPercentagesNormalized)
    predictedClass = CLASSES[classPercentagesNormalized.index(confidenceOfPredicted)]

    return predictedClass, confidenceOfPredicted, classPercentages


#Will not do oneVsAll but perform ONE logistic regression classification.
#returns class(1: right class, 0:wrong class) and confidence
def classifySampleSingleClass(x, ClassWeight):
    currentFeatureVector = getPhi(x)

    wTimesPhi = np.dot(np.transpose(ClassWeight), currentFeatureVector)
   # print(wTimesPhi)
    regressionResult = sigmoid(wTimesPhi)

    if(regressionResult >= 0.5):
        return 1, regressionResult

    return 0, regressionResult


#returns a vector containing as first element the bias b and the others are the features for sample x derived by applying
#the basis function from getBasisOutput on each
def getPhi(x):
    returnVector = np.zeros(len(x)+1)

    for i in range(len(x)+1):
        returnVector[i] = getBasisOutput(x, i)

    return returnVector

def getBasisOutput(x, i):
    if(i == 0):
        return 1
    return x[i-1]

def sigmoid(x):
    return 0.5 * (x/(1+abs(x)) + 1)
    #return 1 / (1 - np.exp(-x))

################
#Learning
################

#Will optimize all the weights for every class. Thereby it does one step for every class and then contiues to the next step.
def optimizeAllWeights(currentWeights, trainingSamples):
    tempWeightsOld = currentWeights

    for i in range(MAX_STEPS):
        tempWeightsOld = currentWeights
        for c in range(len(CLASSES)):
            currentWeights[c] = updateWeightsPerClasStep(tempWeightsOld[c], trainingSamples, CLASSES[c])

        #print(tempWeightsOld)
        #print("Weights: ")
        #print(currentWeights)
        #if(np.array(tempWeightsOld) - np.array(currentWeights) < UPDATE_THRESHOLD * np.ones(len(currentWeights))):
          #  return currentWeights

        currentGeneralError = calcTotalError(currentWeights, trainingSamples)
        print("Progress Global Weight: " + str(i) + " Right: " + str(1-currentGeneralError) + runtime())


    return currentWeights

#Will optimize the weights for one class only. Thereby this will only do one step of gradient decent.
#CurrentWeightsPerClass is the vector contining the weights for this class logistic regression. Training Samples is a list of training samples. Current Class is nominal (string) class value.
def updateWeightsPerClasStep(currentWeightsPerClass, trainingSamples, currentClass):
    newWeights = currentWeightsPerClass
    deltaW = np.zeros(len(currentWeightsPerClass))

    for sample in trainingSamples:
        sampleInput = sample[0]
        sampleTarget = sample[1]
        prediction = classifySampleSingleClass(sampleInput, newWeights)
        target = 0
        if sampleTarget == currentClass:
            target = 1
        for j in range(len(currentWeightsPerClass)):
            #deltaW[j] += abs(prediction[0] - target) * prediction[1] * getBasisOutput(sampleInput, j)
            deltaW[j] += (target - prediction[0]) * getBasisOutput(sampleInput, j)

    #update w with learning rate of its gradient.
    #change1 weights can only be updated with complete gradient
    newWeights = newWeights + deltaW * LEARNING_RATE

    #if(currentClass == "sitting"):
        #print("deltaW")
        #print(deltaW)

        #print "new weights"
        #print(newWeights)
    return newWeights

# #accepts y(currentPrediction) in nominal, t(labeled value) in nominal and classValue(class we optimize for) in nominal (string) form.
# def calculateError(y, t, classValue):
#     if(y == t and y == classValue):
#         return 0
#
#     if(y != classValue and t != classValue):
#         return 0
#
#     return 1

###############
#Helpers
###############
def runtime():
    return " runtime(s): " + str(time.time() - start)

################################
#general control flow section
################################

originalData = readData("../data/dataset-complete_90PercentTrainingSet_mini10Percent.arff")
print("Test reading: " + str(originalData[0]))

zeroWeights = []
for i in range(len(CLASSES)):
    dummyWeight = np.zeros(17)
    zeroWeights.append(dummyWeight)

optimizedWeights = optimizeAllWeights(zeroWeights,originalData)
currentError = calcTotalError(optimizedWeights,originalData)

print("Current Error on Training: " + str(currentError) + runtime())

print("____________________________")
print(optimizedWeights)