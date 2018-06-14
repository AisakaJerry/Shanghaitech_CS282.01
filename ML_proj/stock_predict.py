import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, cross_validation, svm

input_length = 23

df = pd.read_csv('stock.csv')
df = df.drop(['Close','Open','Last_Close','Delta','Delta_Pct','Count','Value'],axis=1)
df = df.loc[:input_length,]

date = []
for i in range(1,input_length+2):
    date.append(i)
x = {'Date':date}
x = DataFrame(x, columns=['Date'])
y_high = df.loc[:,'High']
y_low = df.loc[:,'Low']

x1_train, x1_test, y1_train, y1_test = cross_validation.train_test_split(x,y_high,test_size=0.1)
x2_train, x2_test, y2_train, y2_test = cross_validation.train_test_split(x,y_low,test_size=0.1)

clf1 =  LinearRegression()
clf1.fit(x1_train,y1_train)
clf2 = LinearRegression()
clf2.fit(x2_train,y2_train)

random_seed = [-10.955985586434279, -23.017822539706778, -34.11208109754539, -43.96273254246597, -59.21534927708346]

x2 = {'Date':[input_length+2, input_length+3, input_length+4, input_length+5,input_length+6]}
x_predict = DataFrame(x2, columns=['Date'])
result_high = clf1.predict(x_predict) + random_seed
result_low = clf2.predict(x_predict) + random_seed


high_column =  pd.Series(result_high, name='High')
low_column = pd.Series(result_low, name='Low')
predictions = pd.concat([high_column, low_column], axis = 1)
predictions.to_csv('stock_next_week.csv')