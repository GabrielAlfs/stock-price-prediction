import pandas as pd
import math, datetime
import numpy as np
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression, Lasso, BayesianRidge
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

data = pd.read_csv('MGLU3.SA.csv') 
df = data.loc[:,['Adj Close','Volume']]
df['HL_PCT'] = (data['High'] - data['Low']) / data['Close'] * 100.0
df['PCT_change'] = (data['Close'] - data['Open']) / data['Open'] * 100.0

forecast_col = 'Adj Close'

df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.1 * len(df)))

df['label'] = df['Adj Close'].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

last_date = df.iloc[-1].name

def plot_prediction(model, model_name):
  forecast_set = model.predict(X_lately)
  
  df['Forecast'] = np.nan

  last_unix = datetime.datetime.fromtimestamp(last_date)
  next_unix = last_unix + datetime.timedelta(days=1)
  for i in forecast_set:
    next_date = next_unix
    next_unix += datetime.timedelta(days=1)
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

  df['Adj Close'].plot()
  df['Forecast'].plot()
  
  plt.title(model_name)
  plt.legend(loc=4)
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.show()

# Ordinary Least Squares
lm_lr = LinearRegression()
lm_lr.fit(X_train, y_train)
print(f'Ordinary Least Squares confidence level: {lm_lr.score(X_test, y_test)}')
plot_prediction(lm_lr, 'Ordinary Least Squares')

# Lasso
lm_lasso = Lasso()
lm_lasso.fit(X_train, y_train)
print(f'Lasso confidence level: {lm_lasso.score(X_test, y_test)}')
plot_prediction(lm_lasso, 'Lasso')

# Bayesian Ridge Regression
lm_brb = BayesianRidge()
lm_brb.fit(X_train, y_train)
print(f'Bayesian Ridge Regression confidence level: {lm_brb.score(X_test, y_test)}')
plot_prediction(lm_brb, 'Bayesian Ridge Regression')





