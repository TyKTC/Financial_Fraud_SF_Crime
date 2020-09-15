# -*- coding: utf-8 -*-
"""
Created on Fri May  1 11:27:10 2020

@author: KTC
"""


import numpy as np
import pandas as pd
import toad
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_rows', 250)

#Read original data
train_ori = pd.read_csv('train.csv')
test_ori = pd.read_csv('test.csv')
train_target_ori = pd.read_csv('train_target.csv')

#Combine two data sets
data = pd.merge(train_target_ori, train_ori, on = 'id')

#Exploratary data analytics
exploratary_data = toad.detector.detect(data)

#data describe
data_describe = data.describe()

#Fill nan
data.fillna(0, inplace = True)

#gender:0.005022, weekday:0.018600, thus removing both variables
#Filter useless feature
data, drop_lst = toad.selection.select(data, target = 'target', corr = 2, empty = 0.5, iv = 0.02, 
                                               return_drop = True, exclude = ['id', 'certId', 'dist','certValidBegin', 
                                                                              'certBalidStop','bankCard', 'residentAddr', 'x_79'])

#Get certValid range
data['certValidrange'] = data['certBalidStop'] - data['certValidBegin']
data.drop(['certBalidStop','certValidBegin'],axis = 1, inplace = True)

#All id are unique, thus does't provide more info, removing id
data.drop('id', axis = 1, inplace = True)

#In CertId column, only the first two digits are useful
data['certId_dist'] = data['certId'].astype('str').str[0:2].astype('int')
data.drop('certId', axis = 1, inplace = True)

#Drop unnecessary feature
data.drop('dist', axis = 1, inplace = True)

#Check correlation among features
data_corr_ori = data.corr()[(data.corr() > 0.8)]

#Stacking Methods to combine x_0 - x_79
#First layer
x0_x79_train = np.array(data.loc[:,'x_0':'x_79'])
x0_x79_test = np.array(test_ori.loc[:, 'x_0':'x_79'])
real_y = np.array(data.iloc[:, 0])
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from itertools import chain

kf = KFold(n_splits = 5)
clfs = [LogisticRegression(max_iter = 500), RandomForestClassifier(n_estimators = 10, criterion = 'gini'),
       ExtraTreesClassifier(n_estimators = 10 * 2, criterion = 'gini'), GradientBoostingClassifier(n_estimators = 10)]
clfs_name = ['LogisticRegression', 'RandomForestClassifier', 'ExtraTreesClassifier', 'GradientBoostingClassifier']
new_training_data = []
new_test_data = []
print('First layer of combining x_0 to x_79')
for clf, clf_name in zip(clfs, clfs_name):
    print('New Classification Model Start Now:', clf_name)
    sub_k_fold = []
    sub_test_data = []
    for train_index, test_index in kf.split(x0_x79_train):
#        print('\n')
#        print("Train Index: ", train_index)
#        print("Test Index: ", test_index)
        x_train = x0_x79_train[train_index]
        x_test = x0_x79_train[test_index]
        y_train = real_y[train_index]
        clf.fit(x_train, y_train)
        
        k_fold_predict = clf.predict_proba(x_test)[:,1]
        sub_k_fold.append(list(k_fold_predict))
        
        sub_test = clf.predict_proba(x0_x79_test)[:,1]
        sub_test_data.append(list(sub_test))
        
    sub_new_training_data = list(chain.from_iterable(sub_k_fold))
    new_training_data.append(sub_new_training_data)
    
    sub_test_data_avg = np.mean(sub_test_data, axis=0)
    new_test_data.append(list(sub_test_data_avg))

new_training_data = pd.DataFrame(new_training_data)
new_training_data = new_training_data.T
new_training_data.rename(columns={0:'LogisticRegression',1:'RandomForestClassifier', 2:'ExtraTreesClassifier', 3:'GradientBoostingClassifier'},inplace=True)

new_test_data = pd.DataFrame(new_test_data)
new_test_data = new_test_data.T
new_test_data.rename(columns={0:'LogisticRegression',1:'RandomForestClassifier', 2:'ExtraTreesClassifier', 3:'GradientBoostingClassifier'},inplace=True)

#Second Layer
import xgboost as xgb
print('Second layer of combining x_0 to x_79')
print('XGBoost begin now!!!')
params={'booster':'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth':4,
    'lambda':10,
    'subsample':0.75,
    'colsample_bytree':0.75,
    'min_child_weight':2,
    'eta': 0.025,
    'seed':0,
    'nthread':8,
     'silent':1}

plst = params.items()
dtrain = xgb.DMatrix(new_training_data, real_y)
model = xgb.train(plst, dtrain, 500)

x0_x79 = model.predict(xgb.DMatrix(new_training_data))

data['x0_x79'] = x0_x79
data.drop(data.columns[7: 87], axis = 1, inplace = True)

x0_x79 = model.predict(xgb.DMatrix(new_test_data))

test_ori['x0_x79'] = x0_x79
test_ori.drop(test_ori.columns[10: 90], axis = 1, inplace = True)
print('XGBoost completed')



#Determine cut points
combiner = toad.transform.Combiner()
combiner.fit(data,y='target',method='chi')
bins = combiner.export()
data_bins = combiner.transform(data)

#Checking monotonicity
from toad.plot import bin_plot,badrate_plot
#bin_plot(data_bins,x='x0_x79',target='target')


#Adjust age
adj_bin = {'age':[18.5,42.5,48,52.5]}
combiner.set_rules(adj_bin)
data_bins = combiner.transform(data)
#bin_plot(data_bins,x='age',target='target')

#Adjust age
adj_bin = {'edu':[10]}
combiner.set_rules(adj_bin)
data_bins = combiner.transform(data)
#bin_plot(data_bins,x='edu',target='target')

#Adjust job
adj_bin = {'job':[2,3]}
combiner.set_rules(adj_bin)
data_bins = combiner.transform(data)
#bin_plot(data_bins,x='job',target='target')

#Adjust lmt
adj_bin = {'lmt':[1.3, 2.296, 3.233, 7.267, 9.732999999999999,17.963]}
combiner.set_rules(adj_bin)
data_bins = combiner.transform(data)
#bin_plot(data_bins,x='lmt',target='target')

#Adjust basicLevel
adj_bin = {'basicLevel':[1,2]}
combiner.set_rules(adj_bin)
data_bins = combiner.transform(data)
#bin_plot(data_bins,x='basicLevel',target='target')

#Adjust bankCard
adj_bin = {'bankCard':[0, 405512760, 621673386, 621797589, 622845026, 623052335]}
combiner.set_rules(adj_bin)
data_bins = combiner.transform(data)
#bin_plot(data_bins,x='bankCard',target='target')

#Adjust ethnic
adj_bin = {'ethnic':[1,24]}
combiner.set_rules(adj_bin)
data_bins = combiner.transform(data)
#bin_plot(data_bins,x='ethnic',target='target')

#Adjust residentAddr
adj_bin = {'residentAddr':[650627, 841103]}
combiner.set_rules(adj_bin)
data_bins = combiner.transform(data)
#bin_plot(data_bins,x='residentAddr',target='target')

#Adjust highestEdu
adj_bin = {'highestEdu':[50, 70]}
combiner.set_rules(adj_bin)
data_bins = combiner.transform(data)
#bin_plot(data_bins,x='highestEdu',target='target')

#Adjust linkRela Still have problem: What's the meaning of -999 in Dataset
adj_bin = {'linkRela':[0,10]}
combiner.set_rules(adj_bin)
data_bins = combiner.transform(data)
#bin_plot(data_bins,x='linkRela',target='target')

#Adjust setupHour
adj_bin = {'setupHour':[6,11,17,20]}
combiner.set_rules(adj_bin)
data_bins = combiner.transform(data)
#bin_plot(data_bins,x='setupHour',target='target')

#Adjust certValidrange
adj_bin = {'certValidrange':[315532800, 631238400, 252289638400, 252448268800]}
combiner.set_rules(adj_bin)
data_bins = combiner.transform(data)
#bin_plot(data_bins,x='certValidrange',target='target')

#Adjust certId_dist
adj_bin = {'certId_dist':[34, 56, 57, 65]}
combiner.set_rules(adj_bin)
data_bins = combiner.transform(data)
#bin_plot(data_bins,x='certId_dist',target='target')

#Adjust x0_x79
adj_bin = {'x0_x79':[0.0023,0.0029, 0.0039, 0.0056]}
combiner.set_rules(adj_bin)
data_bins = combiner.transform(data)
#bin_plot(data_bins,x='x0_x79',target='target')

#WOE mapping
t = toad.transform.WOETransformer()
data_woe = t.fit_transform(data_bins,data_bins['target'], exclude = 'target')

#K-fold cross validation
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from toad.metrics import KS, AUC
from sklearn.metrics import confusion_matrix

scores = []
proba_result = []
kf = KFold(n_splits=8)
X = np.array(data_woe.iloc[:, 1:])
y = np.array(data_woe.iloc[:, 0])
for train_index, test_index in kf.split(X):
    print('\n')
    print("Train Index: ", train_index)
    print("Test Index: ", test_index)
    X_train = X[train_index]
    X_test = X[test_index]
    y_train = y[train_index]
    y_test = y[test_index]
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    scores.append(log_reg.score(X_test, y_test))
    pre_proba = log_reg.predict_proba(X_test)[:,1]
    pre_proba_more_than = np.where(pre_proba > 0.01, 1, 0) #The probability of samples becoming target more than 0.01 are marked as 1
    
    #KS and AUC
    print('KS:', KS(pre_proba, y_test))
    print('-------------------------------------------------')
    print('AUC:', AUC(pre_proba, y_test))
    
    #Confusion Matrix
    confu_matrix = confusion_matrix(y_test, pre_proba_more_than)
    print('-------------------------------------------------')
    print('Confusion Matrix:\n',confu_matrix)
    print('-------------------------------------------------')
    print('True Positive Rate:', confu_matrix[1,1] / (confu_matrix[1,0] + confu_matrix[1,1]))
    print('False Positive Rate:',confu_matrix[0,1] / (confu_matrix[0,0] + confu_matrix[0,1]))
    print('-------------------------------------------------')
    print('Recall Ratio:', confu_matrix[1,1] / (confu_matrix[1,1] + confu_matrix[1,0]))
    print('Precision:', confu_matrix[1,1] / (confu_matrix[1,1] + confu_matrix[0,1]))
    
    #ROC curve
    toad.plot.roc_plot(pre_proba_more_than, y_test)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    