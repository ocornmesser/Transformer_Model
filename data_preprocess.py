import pandas as pd
import numpy as np
import csv

n_steps = 48
n_features = 1

# Function to split the sequence
def split_sequence(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i : (end_ix - 1)], sequence[end_ix - 1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Read and process data
with open('DataByHourWithWeather.csv', 'r') as incomingData:
    readFile = csv.reader(incomingData)
    strictData, dayData = [], []
    dayType, dayCounter, pastDay = 0, 4, 26
    for row in readFile:
        if row[5] == 'Demand (kWh)':
            continue
        if row[1] != pastDay: 
            if dayCounter in {6, 7}:
                dayType = 1 if dayCounter == 7 else 0
                dayCounter = 1 if dayCounter == 7 else dayCounter + 1
            else:
                dayType = 0
                dayCounter += 1
        pastDay = row[1]
        strictData.append([float(row[5])])
        dayData.append([row[1], row[2], row[3], row[4]])

scaled_data = strictData

trainData = scaled_data[:int(len(scaled_data) // 1.33)]
testData = scaled_data[int(-len(scaled_data) // 1.33):]
    
XTrain, yTrainWithDay = split_sequence(trainData, n_steps + 1)
XTest, yTestWithDay = split_sequence(testData, n_steps + 1)
yTest = yTestWithDay[:, 0]
yTrain = yTrainWithDay[:, 0]

XTrain = XTrain.reshape(XTrain.shape[0], XTrain.shape[1], n_features)