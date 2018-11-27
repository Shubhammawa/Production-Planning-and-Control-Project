import numpy as np 
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA 
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

series = pd.read_csv("JaipurRawData3.csv").values

# print(series[:,1])
# plt.plot(series[:,12])
# plt.show()

X = series[:,1:12]
Y = series[:,12]
#print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y)

clf = LogisticRegression()
clf.fit(X_train,Y_train)
Y_pred = clf.predict(X_test)

print(metrics.accuracy_score(Y_test,Y_pred))

# print(adfuller(X))
# autocorrelation_plot(X)
# plt.show()
size = int(len(X)*0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(1,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()

# Parameter space
# (2,1,1): MSE = 1.454
# (2,1,0): MSE = 1.441
# (2,0,0): MSE = 1.471
# (2,0,1): MSE = 1.437
