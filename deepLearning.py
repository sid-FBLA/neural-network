import math
import numpy as np
import pandas_datareader as web
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

#stock quote
quote = web.DataReader('TSLA', data_source='yahoo', start='2012-01-01', end="2021-01-01")

print(quote)
print(quote.shape)

plt.figure(figsize=(16,8))
plt.plot(quote['Close'])
plt.xlabel('Date')
plt.ylabel('Price')
#plt.show()

#Dataframe with only Close
data = quote.filter(['Close'])
#Conver to Numpy array
dataset = data.values
#Get # of rows to train model on
training_data_len = len(dataset)

print(training_data_len);

#Scale data to present to Neural Network --> Normalization,
#transforms data to values between 0 and 1
scaler = MinMaxScaler(feature_range=(0,1))
#hold new "scaled dataset"
scaled_data = scaler.fit_transform(dataset)
print(scaled_data)

#create the "training data set"
#Create the scaled trained dataset
train_data = scaled_data[0:training_data_len, :]
print(train_data)
#Split the data into x_train and y_train
#x_train, all values prior to 'y'
x_train = []
#y_train is the value we want the model to predict
y_train = []

#starts from 70 --> length of dataset
for i in range(70, training_data_len):
    #appends all data in x_train array, excluding 'i'
    x_train.append(train_data[i-70:i, 0])
    y_train.append(train_data[i, 0])
    if i <= 70:
        print(i);
        print(x_train)
        print(y_train)
