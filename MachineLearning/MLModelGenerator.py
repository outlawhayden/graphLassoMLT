import scipy
import numpy as np
from scipy.stats import norm
import time
import string
import pandas as pd
import pathlib
from pathlib import Path
import matplotlib.pyplot as plt

import seaborn as sns
import os
from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


#assign project id
letters = string.digits
stringID = str(np.random.randint(10000,99999))

#create new folder for output
folderDir = os.path.join(os.getcwd(),stringID + "Analysis")
Path(folderDir).mkdir(parents=False, exist_ok=False)

#create data and analysis output files
dataName = (stringID + "BulkIntegrate.txt")
analysisName = (stringID + "DataAnalysis.txt")
dataFile = os.path.join(folderDir, dataName)
analysisFile = os.path.join(folderDir, analysisName)

#open files
dp = open(dataFile, 'w')
dp.write("Alpha,Probability\n")
ap = open(analysisFile, 'w')    
ap.write(stringID + " Analysis: \n\n")
ap.close()

#output location of new folder
print("Generation and Analysis saved to: " + folderDir)


dimensions = int(input("Enter number of dimensions to sample from (Integer > 2): "))
iteration = int(input("Enter number of iterations to converge (positive integer): "))
peralpha = int(input("Enter number of iterations per alpha (positive integer): "))
startalpha = float(input("Enter alpha start value (positive float): "))
stepalpha = float(input("Enter alpha step value (positive float): "))
endalpha = float(input("Enter alpha end value (positive float > startalpha): "))
dataportion = float(input("Enter % of data to be used in ml model training set: (float between 0 and 1 non-inclusive)"))



TCoordinates = []

#calculate number of diagonal entries in n matrix
def quantityDiagonalEntries(n):
    n -= 1
    numElements = 0
    while n >0:
        numElements += n
        n -= 1
    return numElements

#generate binary numbers from 1 to n, creates new array
def binaryGenerate(n): ##non-recursive!! use python cast to binary
    localBinaryNums=[];
    N = int((n-1)*n/2)
    for i in range(2**N):
        tempB = format(i,"b")
        localBinaryNums.append(tempB)
    return localBinaryNums;
        
#generate coordinates for n off-diagonal entries in n dimensions
def coordinateGenerate(n): ##works
    s = 0
    for i in range(s,n):
        for j in range (s+1,n):
            TCoordinates.append(np.array([i,j]))
            
        s += 1;
        
        
diagonalEntries = quantityDiagonalEntries(dimensions)
binaryIndices = binaryGenerate(dimensions)             
coordinateGenerate(dimensions)

#create basic t matrix
TBase = np.ones(dimensions, dtype = int) - np.identity(dimensions, dtype=int)    

#create list of t matrices
TList = []      
for b in binaryIndices:
    TTemp = np.ones(dimensions, dtype = int) - np.identity(dimensions, dtype=int)
    for d in range(0,len(b)):
        if (b[d] == "1"):
            TTemp[TCoordinates[d][0]][TCoordinates[d][1]] *= -1
            TTemp[TCoordinates[d][1]][TCoordinates[d][0]] *= -1
    TList.append(TTemp)
    

print(TList)
print("Done")
## for n dim, take n-1 samples


#assign covariance matrix as identity in n dimensions
cov = np.identity(dimensions)
print("Done")

#counter initialization
total=0
hits = 0
ratio = 0
old = 0
new = 0
change = 0
printsamples = 'n'

#monte-carlo sampling

for d in range(3, dimensions):
    cov = np.identity(d)
    for a in np.arange(startalpha, endalpha, stepalpha):
        alpha = a
        for j in range(peralpha):
            #generate data
            for i in range(1,iteration+1):
                old = ratio
                posdef = False
                zerovect = np.zeros(d)
                samples = np.random.multivariate_normal(zerovect, cov, size=2, check_valid='warn', tol=1e-8)
                #if verbose == 'y':
                    #print(samples)
                S = np.dot(samples.T, samples)
                #print(S)
                #check each T
            for t in TList:
                    aT = alpha * t
                    SaT = np.add(S, aT)
                    if np.all(np.linalg.eigvals(SaT) > 0):
                        posdef = True

            if (posdef == True):
                hits += 1

            total += 1
            ratio = hits/total
            new = ratio

            change = new - old

            dp.write(str(d) + "," + str(samples) + "," + str(alpha)+ ","+ str(ratio)+ "\n")

#close data file        
dp.close()
print("Done")