import pandas as pd
import numpy as np
from pandas import *
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import neighbors
from sklearn import cross_validation
from sklearn.preprocessing import Imputer
from math import isnan

df = pd.read_csv('DataTraining.csv')

y1_train = df.loc[:,'responded']
y2_train = df.loc[:,'profit']
df = df.drop(['responded','profit', 'id'], axis = 1)

'''
dummy_fields = ['profession','marital','schooling','default','housing',
                'loan','contact','month','poutcome']
for each in dummy_fields:
    dummies = pd.get_dummies(df.loc[:,each], prefix=each)
    df = pd.concat([df,dummies],axis = 1)
fields_to_drop = ['profession','marital','schooling','default','housing',
                  'loan','contact','month','day_of_week','poutcome']
x_train = df.drop(fields_to_drop, axis = 1)
'''

for name in ['profession','marital','schooling','default','housing', 'loan','contact','month','day_of_week','poutcome']:
    col = pd.Categorical.from_array(df[name])
    df[name] = col.codes
x_train = df

imp = Imputer(missing_values='NaN', strategy='median', axis=0, verbose=0)
imp=imp.fit(x_train)
x_train_imp = imp.transform(x_train)  # deal with NaN value
print(x_train_imp)

y2_train_new=[]
for item in y2_train:
    if isnan(item):
        y2_train_new.append(0)
    else:
        y2_train_new.append(item)
#print(y2_train_new)

clf = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=10, min_samples_split=5, max_features='sqrt', random_state=10)
#clf = neighbors.KNeighborsClassifier()
clf.fit(x_train_imp, y1_train)
clf2 = neighbors.KNeighborsRegressor()
clf2.fit(x_train_imp, y2_train_new)

# cross_validation
print('clf score=',sum(cross_validation.cross_val_score(clf, x_train_imp, y1_train, cv=10))/10)
print('clf2 score=',sum(cross_validation.cross_val_score(clf2, x_train_imp, y2_train_new, cv=10))/10)

# decision tree classifier complete, start to predict
df2 = pd.read_csv('DataPredict.csv',header = None)
df2.columns=['custAge','profession','marital','schooling','default','housing','loan',
                         'contact','month','day_of_week','campaign','pdays','previous','poutcome',
                         'emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed',
                         'pmonths','pastEmail']

for name in ['profession','marital','schooling','default','housing', 'loan','contact','month','day_of_week','poutcome']:
    col = pd.Categorical.from_array(df2[name])
    df2[name] = col.codes
x_test = df2
x_test_imp = imp.transform(x_test)
#print(x_test)

result_respond = clf.predict(x_test_imp)
result_profit = clf2.predict(x_test_imp)

count = 0
tot_prof = 0
result_market = []
for i in range(len(result_profit)):
    if result_profit[i]>30 and result_respond[i]=='yes':
        count += 1
        result_market.append('yes')
        tot_prof += (result_profit[i] - 30)
    else:
        result_market.append('no')

print(count)
print('tot_prof=', tot_prof)
df_out = pd.DataFrame()
respond_column = pd.Series(result_respond, name='respond')
profit_column = pd.Series(result_profit, name='profit')
market_column = pd.Series(result_market, name='final_market')
predictions = pd.concat([respond_column, profit_column, market_column], axis = 1)
predictions.to_csv('testingCandidate_Reg.csv')


