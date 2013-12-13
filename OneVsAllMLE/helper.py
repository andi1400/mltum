import numpy as np
import csv
import time

class helper():
    def calcTotalError(self, classifier, trainingSamples, currentWeights):
        curRight = 0
        confusionMatrix = []
        for i in range(len(classifier.CLASSES)):
            confusionMatrix.append([])
            for j in range(len(classifier.CLASSES)):
                confusionMatrix[i].append(0)


        for i in range(len(trainingSamples)):
            result = classifier.classifySample(trainingSamples[i][0], currentWeights)[0]
            target = trainingSamples[i][1]
            if (result == target):
                curRight += 1
            confusionMatrix[classifier.CLASSES.index(result)][classifier.CLASSES.index(target)] += 1

        return 1-float(curRight)/len(trainingSamples), confusionMatrix

    def writeWeightsDebug(self, filename, weights):
        np.savetxt(filename, weights, delimiter=";")

    #read the weights from csv and turn them into the weight vecotr.
    def readWeights(self, filename, classes):
        weights = np.genfromtxt(filename, delimiter=';')

        print("Successfully read weight vector: ")
        print(weights)

        return weights

    def writeWeights(self, filename, classes, weights, writeClassNames):
        with open(filename, "a") as file:
            csvwriter = csv.writer(file, delimiter=";", quotechar='"')

            for i in range(len(classes)):
                if writeClassNames:
                    csvwriter.writerow([classes[i]] + weights[i].tolist())
                else:
                    csvwriter.writerow(weights[i].tolist())

    #returns a vector containing as first element the bias b and the others are the features for sample x derived by applying
    #the basis function from getBasisOutput on each
    def getPhi(self, x, basisFunction):
        returnVector = np.zeros(len(x)+1)

        for i in range(len(x)+1):
            returnVector[i] = basisFunction(self, x, i)

        return returnVector

    def getXDirectly(self, x, i):
        if(i == 0):
            return 1
        return x[i-1]

    def pseudoSigmoid(self, x):
        return 0.5 * (x/(1+abs(x)) + 1)

    def sigmoid(self, x):
        return 1 / (1 - np.exp(-x))

    def strRuntime(self, starttime):
        return " runtime(s): " + str(time.time() - starttime)