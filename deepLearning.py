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
quote = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end="2021-01-01")

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
training_data_len = math.ceil(len(dataset) * .8)

print(training_data_len);

#Scale data to present to Neural Network --> Normalization,
#transforms data to values between 0 and 1
scaler = MinMaxScaler(feature_range=(0,1))
#hold new "scaled dataset"
scaled_data = scaler.fit_transform(dataset)

#create the "training data set"
#Create the scaled trained dataset
train_data = scaled_data[0:training_data_len, :]
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


#convert x_train and y_train to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)

#Reshape the data
#LSTM Network expects 3D input
#, # of samples (rows), # of time steps (columns), # of Features (close option), current set is 2D
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

#Compiling model
model.compile(optimizer='adam', loss='mean_squared_error')

#train the model fit = train
#epoch # of iterations when dataset is passed forward and backwards through network
model.fit(x_train, y_train, batch_size=1, epochs=1)

#Create the testing dataset
#Create a new array from index 1742 to 2265
test_data = scaled_data[training_data_len - 70: , :]
print(len(test_data))
#create data sets x_test and y_test
x_test = []
#y_test = all values we want model to predict
y_test = dataset[training_data_len:, :]
for i in range (70, len(test_data)):
    x_test.append(test_data[i-70:i, 0])

#Convert data to numpy arrays to use in LSTM
x_test = np.array(x_test)
#Reshape to 3D for LSTM Model
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#get the prediction values
predictions = model.predict(x_test)
#inverse transform ("unscales values") should be the same as y-values
predictions = scaler.inverse_transform(predictions)

#Get the root mean squared error -- standard deviation of residual (** = ^)
rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
print(rmse)

#Plotting data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

#Visualize model compares actual data to predicted
plt.figure(figsize=(16,8))
plt.title('Predictions Model')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.plot(train['Close'])
plt.plot(valid[['Close',  'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right');
plt.show()

endDate = "2021-01-"
endDateDay = 4
if (endDateDay < 10) :
    endDateDay = '0' + str(endDateDay)
endDateFirst = endDate + str(endDateDay)
endDateDay = endDateDay + 1;
if (endDateDay < 10) :
    endDateDay = '0' + endDateDay
endDateLast = endDate + str(endDateDay)
print(endDateFirst)
print(endDateLast)
#show valid and predicted prices
#Get quote for predictions furhter into the future
apple_quote = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end=endDate)
#New dataframe
new_df = apple_quote.filter(['Close'])
#Get the last 70 days and convert dataframe to array
last_60_days = new_df[-60:].values
#Scale data, no need for fit_transform as we have same MinMax
last_60_days_scaled = scaler.transform(last_60_days)
#Create empty list
X_test = []
#Append past 60 days to list
X_test.append(last_60_days_scaled)
#Convert X_test to a numpy array
X_test = np.array(X_test)
#Reshape
x_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#Getting predicted scale price
pred_price = model.predict(X_test)
#undo scaling
pred_price = scaler.inverse_transform(pred_price)
#predicted price
print("predicted price")
print(pred_price)
#get actual price
apple_quote2 = web.DataReader('AAPL', data_source='yahoo', start='2021-01-05', end="2021-01-5")
print(apple_quote2['Close'])
apple_quote3 = web.DataReader('AAPL', data_source='yahoo', start=endDate, end=endDate)
print(apple_quote3['Close'])
