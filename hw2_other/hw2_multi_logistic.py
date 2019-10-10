import pandas as pd
import numpy as np

import random

class Softmax:
    def softmax(self, X):
        X = X - np.max(X)
        return np.exp(X) / (np.sum(np.exp(X)))
    
    def train(self, digits, labels, maxIter = 10, alpha = 0.1, batchSize = 20):
        self.weights = np.random.uniform(0, 1, (10, 784))
        for iter in range(maxIter):
            batchIndex = self.rand(batchSize)
            for i in batchIndex:
                x = digits[i].reshape(-1, 1)
                y = np.zeros((10, 1))
                y[labels[i]] = 1
                y_ = self.softmax(np.dot(self.weights, x))
                self.weights -= alpha * (np.dot((y_ - y), x.T))
        return self.weights

    def predict(self, digit):   
        return np.argmax(np.dot(self.weights, digit))  
    
    def rand(self, num):
        res = []
        for j in range(num):
            res.append(random.randint(0,59999))
            return res
if __name__ == '__main__':
    
    data_test = pd.read_csv("mnist_test.csv",header = None) .values
    data_train = pd.read_csv("mnist_train.csv",header = None).values

    testDigits = data_test[:,1:]
    testLabels = data_test[:,0]
    trainDigits = data_train[:,1:]
    trainLabels = data_train[:,0]
    
    softmax = Softmax()
    softmax.train(trainDigits, trainLabels, maxIter=10000, batchSize = 20) 

    accuracy = 0
    N = len(testDigits) 
    confusionMatrix = np.zeros((10,10))
    for i in range(N):
        digit = testDigits[i]   
        label = testLabels[i]   
        predict = softmax.predict(digit) 
        if (predict == label):
            accuracy += 1
        confusionMatrix[predict, label]+=1
        print("predict:%d, actual:%d"% (predict, label))
    for i in range(10):
        for j in range(10):
            confusionMatrix[i,j] = np.int(confusionMatrix[i,j])
    print('{:f}'.format,confusionMatrix)
    print("accuracy:%.1f%%" %(accuracy / N * 100))
