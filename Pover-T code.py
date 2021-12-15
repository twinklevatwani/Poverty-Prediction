import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

os.chdir('F:\\Hackathon\\2. Pover-T Tests')

A_hhold_train = pd.read_csv("A_hhold_train.csv")
B_hhold_train = pd.read_csv("B_hhold_train.csv")
C_hhold_train = pd.read_csv("C_hhold_train.csv")
A_indiv_train = pd.read_csv("A_indiv_train.csv")
B_indiv_train = pd.read_csv("B_indiv_train.csv")
C_indiv_train = pd.read_csv("C_indiv_train.csv")
A_hhold_test = pd.read_csv("A_hhold_test.csv")
B_hhold_test = pd.read_csv("B_hhold_test.csv")
C_hhold_test = pd.read_csv("C_hhold_test.csv")
A_indiv_test = pd.read_csv("A_indiv_test.csv")
B_indiv_test = pd.read_csv("B_indiv_test.csv")
C_indiv_test = pd.read_csv("C_indiv_test.csv")

## Country A
# train
# Separating categorical and numerical of individual data for aggregation purpose.

col_names_1 = list()
for i in range(len(A_indiv_train.columns)):
    if(A_indiv_train.iloc[:,i].dtype == 'O' or A_indiv_train.iloc[:,i].dtype == 'bool'):
        col_names_1.append(A_indiv_train.columns[i])

col_names_1.remove('country')
temp_cat = A_indiv_train[col_names_1]
temp_cat['id'] = A_indiv_train['id']
temp_num = A_indiv_train[list(set(A_indiv_train)-(set(col_names_1)))]
temp_num = temp_num.groupby('id').agg({'iid' : 'count' ,'ukWqmeSS' : 'mean','OdXpbPGJ' : 'mean'})
temp_cat = temp_cat.groupby('id').agg(lambda x:x.value_counts().index[0])
A_indiv_df = pd.concat([temp_cat,temp_num],axis = 1)

ind = A_indiv_df.index.values
A_indiv_df['id'] = ind

## Dropping column 'poor' from A_indiv_df, to avoid duplicating poor columns while merging.
A_indiv_df.drop('poor', inplace = True, axis = 1)
A_df = pd.merge(A_hhold_train, A_indiv_df, on = 'id')

## To separate categorical and numerical columns for converting into factor variables.
col_names_2 = list()
for i in range(len(A_df.columns)):
    if (A_df.iloc[:, i].dtype == 'O'):
        col_names_2.append(A_df.columns[i])

col_names_2.remove('country')
temp_cat = A_df[col_names_2]
temp_num = A_df[list(set(A_df) - (set(col_names_2)))]
temp_cat = temp_cat.apply(LabelEncoder().fit_transform)

A_df = pd.concat([temp_cat, temp_num], axis=1)
A_df_x = A_df.drop(['poor', 'country'], axis=1)
A_df_y = A_df['poor']

## Data manipulation on Test data

col_names_1.remove('poor')
temp_cat = A_indiv_test[col_names_1]
temp_cat['id'] = A_indiv_test['id']
temp_num = A_indiv_test[list(set(A_indiv_test) - (set(col_names_1)))]
temp_num = temp_num.groupby('id').agg({'iid': 'count', 'ukWqmeSS': 'mean', 'OdXpbPGJ': 'mean'})
temp_cat = temp_cat.groupby('id').agg(lambda x: x.value_counts().index[0])
A_indiv_testdf = pd.concat([temp_cat, temp_num], axis=1)

ind = A_indiv_testdf.index.values
A_indiv_testdf['id'] = ind
A_test = pd.merge(A_hhold_test, A_indiv_testdf, on='id')

## To separate categorical and numerical columns for converting into factor variables.

temp_cat = A_test[col_names_2]
temp_num = A_test[list(set(A_test) - (set(col_names_2)))]
temp_cat = temp_cat.apply(LabelEncoder().fit_transform)
A_test = pd.concat([temp_cat, temp_num], axis=1)
A_test.drop('country', axis=1, inplace=True)

## Logistic Regression
model = LogisticRegression()
model.fit(A_df_x,A_df_y)

result = model.predict_proba(A_test)
accuracy = accuracy_score(A_df_y, model.predict(A_df_x))
rocauc= roc_auc_score(A_df_y, model.predict(A_df_x))

print(accuracy)
print(rocauc)

# resut[0] = prob of False(0) and result[1] = prob of True(1)
result[:,1]
result_A_lr = pd.DataFrame({ 'id': A_test.id,'country' : A_hhold_test.country, 'poor': result[:,1]})

## RandomForest
modelrf = RandomForestClassifier()
modelrf.fit(A_df_x,A_df_y)

result = modelrf.predict_proba(A_test)
accuracy = accuracy_score(A_df_y, modelrf.predict(A_df_x))
rocauc= roc_auc_score(A_df_y, modelrf.predict(A_df_x))

print(accuracy)
print(rocauc)

result_A_rf = pd.DataFrame({ 'id': A_test.id,'country' : A_hhold_test.country, 'poor': result[:,1]})

## XGBoost
modelxg = XGBClassifier()
modelxg.fit(A_df_x,A_df_y)

result = modelxg.predict_proba(A_test)
accuracy = accuracy_score(A_df_y, modelxg.predict(A_df_x))
rocauc= roc_auc_score(A_df_y, modelxg.predict(A_df_x))

print(accuracy)
print(rocauc)

result_A_xg = pd.DataFrame({ 'id': A_test.id,'country' : A_hhold_test.country, 'poor': result[:,1]})


## Country B
# train
# Separating categorical and numerical of individual data for aggregation purpose.

col_names_1 = list()
for i in range(len(B_indiv_train.columns)):
    if(B_indiv_train.iloc[:,i].dtype == 'O' or B_indiv_train.iloc[:,i].dtype == 'bool'):
        col_names_1.append(B_indiv_train.columns[i])

col_names_1.remove('country')
temp_cat = B_indiv_train[col_names_1]
temp_cat['id'] = B_indiv_train['id']
temp_num = B_indiv_train[list(set(B_indiv_train)-(set(col_names_1)))]

temp_iid = B_indiv_train[['id','iid']].groupby('id').count()
temp_num.drop('iid',axis =1, inplace = True)
temp_num = temp_num.groupby('id').mean()
temp_num['iid'] = temp_iid
temp_cat = temp_cat.groupby('id').agg(lambda x:x.value_counts().index[0])

## Dropping the numeric columns, with more than 30% NA values, ie more than 1000 missing values

## Removing the columns
col_names = list()
for i in range(len(temp_num.columns)):
    if(temp_num.iloc[:,i].isnull().sum() > 1000):
        col_names.append(temp_num.columns[i])


temp_num.drop(col_names, axis = 1, inplace = True)


B_indiv_df = pd.concat([temp_cat,temp_num],axis = 1)

ind = B_indiv_df.index.values
B_indiv_df['id'] = ind

## Dropping column 'poor' from A_indiv_df, to avoid duplicating poor columns while merging.
B_indiv_df.drop('poor', inplace = True, axis = 1)
B_df = pd.merge(B_hhold_train, B_indiv_df, on = 'id')

## Imputing the missing values :
for i in range(len(B_df.columns)):
    if(B_df.iloc[:,i].isnull().any() and B_df.iloc[:,i].dtypes in ['int64','float64']):
        B_df.iloc[:,i].fillna(B_df.iloc[:,i].mean(),inplace = True)

## To separate categorical and numerical columns for converting into factor variables.
col_names_2 = list()
for i in range(len(B_df.columns)):
    if (B_df.iloc[:, i].dtype == 'O'):
        col_names_2.append(B_df.columns[i])

col_names_2.remove('country')
temp_cat = B_df[col_names_2]
temp_num = B_df[list(set(B_df) - (set(col_names_2)))]
temp_cat = temp_cat.apply(LabelEncoder().fit_transform)

B_df = pd.concat([temp_cat, temp_num], axis=1)
B_df_x = B_df.drop(['poor', 'country'], axis=1)
B_df_y = B_df['poor']

## Data manipulation on Test data

col_names_1.remove('poor')
temp_cat = B_indiv_test[col_names_1]
temp_cat['id'] = B_indiv_test['id']
temp_num = B_indiv_test[list(set(B_indiv_test) - (set(col_names_1)))]

temp_iid = B_indiv_test[['id', 'iid']].groupby('id').count()
temp_num.drop('iid', axis=1, inplace=True)
temp_num = temp_num.groupby('id').mean()
temp_num['iid'] = temp_iid
temp_cat = temp_cat.groupby('id').agg(lambda x: x.value_counts().index[0])

## Dropping the numeric columns, with more than 30% NA values, that were dropped from train itself
## Removing the columns
temp_num.drop(col_names, axis=1, inplace=True)

B_indiv_testdf = pd.concat([temp_cat, temp_num], axis=1)

ind = B_indiv_testdf.index.values
B_indiv_testdf['id'] = ind

B_test = pd.merge(B_hhold_test, B_indiv_testdf, on='id')

## Imputing the missing values :
for i in range(len(B_test.columns)):
    if (B_test.iloc[:, i].isnull().any() and B_test.iloc[:, i].dtypes in ['int64', 'float64']):
        B_test.iloc[:, i].fillna(B_test.iloc[:, i].mean(), inplace=True)

## To separate categorical and numerical columns for converting into factor variables.

temp_cat = B_test[col_names_2]
temp_num = B_test[list(set(B_test) - (set(col_names_2)))]
temp_cat = temp_cat.apply(LabelEncoder().fit_transform)

B_test = pd.concat([temp_cat, temp_num], axis=1)
B_test.drop('country', axis=1, inplace=True)


## Logistic Regression
model = LogisticRegression()
model.fit(B_df_x,B_df_y)
result = model.predict_proba(B_test)
accuracy = accuracy_score(B_df_y, model.predict(B_df_x))
rocauc= roc_auc_score(B_df_y, model.predict(B_df_x))

print(accuracy)
print(rocauc)

# resut[0] = prob of False(0) and result[1] = prob of True(1)
result[:,1]
result_B_lr = pd.DataFrame({ 'id': B_test.id,'country' : B_hhold_test.country, 'poor': result[:,1]})

## RandomForest
modelrf = RandomForestClassifier()
modelrf.fit(B_df_x,B_df_y)

result = modelrf.predict_proba(B_test)
accuracy = accuracy_score(B_df_y, modelrf.predict(B_df_x))
rocauc= roc_auc_score(B_df_y, modelrf.predict(B_df_x))

print(accuracy)
print(rocauc)

result_B_rf = pd.DataFrame({ 'id': B_test.id,'country' : B_hhold_test.country, 'poor': result[:,1]})


## XGBoost
modelxg = XGBClassifier()
modelxg.fit(B_df_x,B_df_y)

result = modelxg.predict_proba(B_test)
accuracy = accuracy_score(B_df_y, modelxg.predict(B_df_x))
rocauc= roc_auc_score(B_df_y, modelxg.predict(B_df_x))

print(accuracy)
print(rocauc)

result_B_xg = pd.DataFrame({ 'id': B_test.id,'country' : B_hhold_test.country, 'poor': result[:,1]})


## Country C

# train
# Separating categorical and numerical of individual data for aggregation purpose.

col_names_1 = list()
for i in range(len(C_indiv_train.columns)):
    if(C_indiv_train.iloc[:,i].dtype == 'O' or C_indiv_train.iloc[:,i].dtype == 'bool'):
        col_names_1.append(C_indiv_train.columns[i])

col_names_1.remove('country')
temp_cat = C_indiv_train[col_names_1]
temp_cat['id'] = C_indiv_train['id']
temp_num = C_indiv_train[list(set(C_indiv_train)-(set(col_names_1)))]

temp_iid = C_indiv_train[['id','iid']].groupby('id').count()
temp_num.drop('iid',axis =1, inplace = True)
temp_num = temp_num.groupby('id').mean()
temp_num['iid'] = temp_iid
temp_cat = temp_cat.groupby('id').agg(lambda x:x.value_counts().index[0])
C_indiv_df = pd.concat([temp_cat,temp_num],axis = 1)

ind = C_indiv_df.index.values
C_indiv_df['id'] = ind

## Dropping column 'poor' from A_indiv_df, to avoid duplicating poor columns while merging.
C_indiv_df.drop('poor', inplace = True, axis = 1)
C_df = pd.merge(C_hhold_train, C_indiv_df, on = 'id')

## To separate categorical and numerical columns for converting into factor variables.
col_names_2 = list()
for i in range(len(C_df.columns)):
    if (C_df.iloc[:, i].dtype == 'O'):
        col_names_2.append(C_df.columns[i])

col_names_2.remove('country')
temp_cat = C_df[col_names_2]
temp_num = C_df[list(set(C_df) - (set(col_names_2)))]
temp_cat = temp_cat.apply(LabelEncoder().fit_transform)

C_df = pd.concat([temp_cat, temp_num], axis=1)
C_df_x = C_df.drop(['poor', 'country'], axis=1)
C_df_y = C_df['poor']

## Data manipulation on Test data

col_names_1.remove('poor')
temp_cat = C_indiv_test[col_names_1]
temp_cat['id'] = C_indiv_test['id']
temp_num = C_indiv_test[list(set(C_indiv_test) - (set(col_names_1)))]

temp_iid = C_indiv_test[['id', 'iid']].groupby('id').count()
temp_num.drop('iid', axis=1, inplace=True)
temp_num = temp_num.groupby('id').mean()
temp_num['iid'] = temp_iid
temp_cat = temp_cat.groupby('id').agg(lambda x: x.value_counts().index[0])
C_indiv_testdf = pd.concat([temp_cat, temp_num], axis=1)

ind = C_indiv_testdf.index.values
C_indiv_testdf['id'] = ind

C_test = pd.merge(C_hhold_test, C_indiv_testdf, on='id')

## To separate categorical and numerical columns for converting into factor variables.

temp_cat = C_test[col_names_2]
temp_num = C_test[list(set(C_test) - (set(col_names_2)))]
temp_cat = temp_cat.apply(LabelEncoder().fit_transform)

C_test = pd.concat([temp_cat, temp_num], axis=1)
C_test.drop('country', axis=1, inplace=True)


# Logistic Regression
model = LogisticRegression()
model.fit(C_df_x,C_df_y)
result = model.predict_proba(C_test)
accuracy = accuracy_score(C_df_y, model.predict(C_df_x))
rocauc= roc_auc_score(C_df_y, model.predict(C_df_x))

print(accuracy)
print(rocauc)

# resut[0] = prob of False(0) and result[1] = prob of True(1)
result[:,1]
result_C_lr = pd.DataFrame({ 'id': C_test.id,'country' : C_hhold_test.country, 'poor': result[:,1]})


## RandomForest
modelrf = RandomForestClassifier()
modelrf.fit(C_df_x,C_df_y)
result = modelrf.predict_proba(C_test)
accuracy = accuracy_score(C_df_y, modelrf.predict(C_df_x))
rocauc= roc_auc_score(C_df_y, modelrf.predict(C_df_x))

print(accuracy)
print(rocauc)

result_C_rf = pd.DataFrame({ 'id': C_test.id,'country' : C_hhold_test.country, 'poor': result[:,1]})


## XGBoost
modelxg = XGBClassifier()
modelxg.fit(C_df_x,C_df_y)
result = modelxg.predict_proba(C_test)
accuracy = accuracy_score(C_df_y, modelxg.predict(C_df_x))
rocauc= roc_auc_score(C_df_y, modelxg.predict(C_df_x))

print(accuracy)
print(rocauc)

result_C_xg = pd.DataFrame({ 'id': C_test.id,'country' : C_hhold_test.country, 'poor': result[:,1]})


result_final_lr = pd.concat([result_A_lr,result_B_lr,result_C_lr], axis = 0)
result_final_rf = pd.concat([result_A_rf,result_B_rf,result_C_rf], axis = 0)
result_final_xg = pd.concat([result_A_xg,result_B_xg,result_C_xg], axis = 0)
result_final1_lr = result_final[['id','country','poor']]
result_final1_rf = result_final[['id','country','poor']]
result_final1_xg = result_final[['id','country','poor']]
result_final1_lr.to_csv('submission_lr.csv')
result_final1_rf.to_csv('submission_rf.csv')
result_final1_xg.to_csv('submission_xg.csv')







