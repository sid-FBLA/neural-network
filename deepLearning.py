import math
import numpy as np
import pandas_datareader as web
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM

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
#Convert to Numpy array
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

#set variable for length of training set
initial = 70

#starts from 70 --> length of dataset
for i in range(70, training_data_len):
    #appends all data in x_train array, excluding 'i'
    x_train.append(train_data[i-70:i, 0])
    #only gets i
    y_train.append(train_data[i, 0])
    if i <= 71:
        print(i);
        print(x_train)
        print(y_train)


#convert x_train and y_train to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)
print(np.mean(x_train))
print(np.mean(y_train))

#Reshape the data
#LSTM Network expects 3D input
#, # of samples (rows), # of time steps (columns), # of Features (close option), current set is 2D
print(x_train.shape)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)

#Builds LSTM Model
model = Sequential()
#50 Neurons, first layers so return is true, shape is time steps, features
model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1)))
#second layers
model.add(LSTM(50, return_sequences=False))
#adding densely connected neural network w/ 25 neurons
model.add(Dense(25))
model.add(Dense(1))
print(model)
