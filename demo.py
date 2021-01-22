list = [0, 1, 2, 4, 4, 5];
print(list[3]);
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

#read_data
dataframe = pd.read_fwf('brain_body.txt')
x_values = dataframe[['Brains']]
y_values = dataframe[['Body']]

#train model on data
body_reg = linear_model.LinearRegression();
body_reg.fit(x_values, y_values);

#visualize results
plt.scatter(x_values, y_values);
plt.plot(x_values, body_reg.predict(x_values));
print(plt.show());
