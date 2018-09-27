import pandas as pd
import numpy as np 
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt

series = pd.read_csv("IPG2211A2N.csv").values
value = series[:,1]
#sns.lineplot(value)
#plt.show(value)
#value = 1 + np.log(value)
print(value)

model = ARIMA(value, order=(2,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
residuals = DataFrame(model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())
#plt.plot(series)