

from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
import df_prepare as dfp
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot


#%%

from pandas import read_csv
from statsmodels.tsa.arima_model import ARIMA

path_to_data_folder = 'C:/Users/Yann/Documents/EPFL/PDM_git/Code/'
name = 'Montana_Timeseries'
file_format = '.csv'
data = dfp.df_from_time_series2(path_to_data_folder+name+file_format, beg_year = 2018)
print('Data loaded')
series = data['DCPmp']
X = series.values[:10000]
X = X.astype('float32')
train_size = int(0.9 * len(X))
train, test = X[:train_size], X[train_size:]
#%%
X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
conf_low = []
conf_high = []
obs = []
for t in range(24*6,72*6):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    yhat, stderr, conf = model_fit.forecast(alpha = 0.05)
    predictions.append(yhat)
    conf_low.append(conf[0][0])
    conf_high.append(conf[0][1])
    obs.append(test[t])
    history.append(test[t])
    print(test[t])
    print('predicted=%f, expected=%f, Prediction Interval: between %.3f and %.3f'  % (yhat, obs[-1], conf[0][0], conf[0][1]))
# error = mean_squared_error(test, predictions)
# print('Test MSE: %.3f' % error)
# # plot
#%%
import matplotlib.pyplot as plt
plt.plot(obs, label = 'expeced')
plt.plot(predictions, label = 'predicted')
# plt.plot(conf_low, label = 'confidence interval low')
# plt.plot(conf_high, label = 'confidence interval high')
plt.legend()
plt.show()

#%%

# size = len(X) - 1
# train, test = X[0:size], X[size:]
model = ARIMA(train, order=(5,1,1))
model_fit = model.fit(disp=False)
intervals = [0.4, 0.3, 0.2, 0.1, 0.05]
for a in intervals:
    forecast, stderr, conf = model_fit.forecast(alpha=a)
    print('%.1f%% Prediction Interval: %.3f between %.3f and %.3f' % ((1-a)*100, forecast, conf[0][0], conf[0][1]))

model_fit.plot_predict(len(train)-1000, len(train)+100)
import pandas as pd
import matplotlib.pyplot as plt
plt.show()
