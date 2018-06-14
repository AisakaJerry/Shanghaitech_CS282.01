import pandas as pd
from pandas import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import Imputer
from math import isnan


df_train = pd.read_csv('DataTraining.csv')


y1_train = df_train.loc[:,'responded']
y2_train = df_train.loc[:,'profit']

df_train = df_train.drop(['responded','profit', 'id'], axis = 1)

for i in range (0,8137):
    if isnan(df_train.loc[i,'custAge']):
        df_train.loc[i,'custAge'] = 25


dummy_fields = ['custAge','profession','marital','schooling','default','housing',
                'loan','contact','month','day_of_week','poutcome']
for each in dummy_fields:
    dummies = pd.get_dummies(df_train.loc[:,each],prefix = each,prefix_sep="_")
    df_train = pd.concat([df_train,dummies],axis = 1)

# print (df.loc[:,'custAge'])

fields_to_drop = ['custAge','profession','marital','schooling','default','housing',
                  'loan','contact','month','day_of_week','poutcome']
for each in fields_to_drop:
    df_train = df_train.drop([each], axis = 1)

df_train.to_csv('data_to_train.csv')

x_train = df_train
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp = imp.fit(x_train)
x_train_imp = imp.transform(x_train)  # deal with NaN value

y2_train_new=[]
for item in y2_train:
    if isnan(item):
        y2_train_new.append(0)
    else:
        y2_train_new.append(item)
#print(y2_train_new)

clf = RandomForestClassifier(n_estimators=1000, max_depth=20, min_samples_leaf=10, min_samples_split=5, max_features='sqrt', random_state=10)
clf.fit(x_train_imp, y1_train)
clf2 = Ridge(alpha=.5)
clf2.fit(x_train_imp, y2_train_new)

# decision tree classifier complete, start to predict
# df2 = pd.read_csv('DataPredict.csv',header = None)
# df2.columns=['custAge','profession','marital','schooling','default','housing','loan',
#                          'contact','month','day_of_week','campaign','pdays','previous','poutcome',
#                          'emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed',
#                          'pmonths','pastEmail']

# for name in ['profession','marital','schooling','default','housing', 'loan','contact','month','day_of_week','poutcome']:
#     col = pd.Categorical.from_array(df2[name])
#     df2[name] = col.codes

df_test = pd.read_csv('DataTraining.csv')

y1_train = df_test.loc[:,'responded']
y2_train = df_test.loc[:,'profit']



df_test = df_test.drop(['responded','profit', 'id'], axis = 1)

df_test.columns=['custAge','profession','marital','schooling','default','housing','loan',
                         'contact','month','day_of_week','campaign','pdays','previous','poutcome',
                         'emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed',
                         'pmonths','pastEmail']


for i in range (0,8137):
    if isnan(df_train.loc[i,'custAge']):
        df_test.loc[i,'custAge'] = 25


dummy_fields = ['custAge','profession','marital','schooling','default','housing',
                'loan','contact','month','day_of_week','poutcome']
for each in dummy_fields:
    dummies = pd.get_dummies(df_test.loc[:,each],prefix = each,prefix_sep="_")
    # print (dummies)
    df_test = pd.concat([df_test,dummies],axis = 1)

# print (df.loc[:,'custAge'])

fields_to_drop = ['custAge','profession','marital','schooling','default','housing',
                  'loan','contact','month','day_of_week','poutcome']
for each in fields_to_drop:
    df_test = df_test.drop([each], axis = 1)


# print(y2_train.ix[:5])
# print("#####################################")
# print("#")
# print("#####################################")
df_test.to_csv('data_to_test.csv')

x_test = df_test
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
        tot_prof += (result_profit[i]-30)
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


