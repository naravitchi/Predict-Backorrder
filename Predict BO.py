# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 01:01:47 2023

@author: floke
"""

import pandas as pd
import numpy as np
import math
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn import feature_selection
import statsmodels.api as sm
from sklearn.feature_selection import mutual_info_regression  
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf 
import csv
import seaborn as sns
import statsmodels.api as sm
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score,confusion_matrix
from collections import Counter
from xgboost import XGBClassifier
import shap
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import tqdm
import torch
from sklearn.model_selection import train_test_split




os.chdir('C:\\Users\\floke\\Desktop\\Manchester\\Study\\Dissertation')
os.getcwd() 
pd.options.display.float_format = '{:.10f}'.format
pd.set_option('display.max_columns', None)



df_train_1 = pd.read_csv("1st data set\\Training_BOP.csv")
df_test_1 = pd.read_csv("1st data set\\Testing_BOP.csv") 
fraction = len(df_test_1)/len(df_train_1)
print(f"No. of training examples: {df_train_1.shape[0]}")
print(f"No. of testing examples: {df_test_1.shape[0]}")

df_1 = df_train_1.merge(df_test_1,how='outer')
df_1['went_on_backorder'].value_counts()

df_2= pd.read_csv("2nd data set\\bopredict.csv")
df_train_2 = df_2.sample(frac=1-fraction, random_state=25)
df_test_2 = df_2.drop(df_train_2.index)

print(f"No. of training examples: {df_train_2.shape[0]}")
print(f"No. of testing examples: {df_test_2.shape[0]}")

#check nan

df_1.columns
df_2.columns
df_1.isna().any()
df_1.isna().sum()

df_2.isna().any()
df_2.isna().sum()

df_1.describe()
df_2.describe()


#check  missing values
print(f" 'percent nan of lead time in df_1:' {df_1.isna().sum()['lead_time']/len(df_1)}")
print(f" 'percent nan of perf_6_month_avg in df_1:' {len(df_1['perf_6_month_avg'][df_1['perf_6_month_avg']==-99])/len(df_1)}")
print(f" 'percent nan of perf_12_month_avg in df_1:' {len(df_1['perf_12_month_avg'][df_1['perf_12_month_avg']==-99])/len(df_1)}")



#fill nan with median
imputer = SimpleImputer(strategy='median', missing_values=np.nan)
df_1['lead_time'] = imputer.fit_transform(df_1['lead_time'].values.reshape(-1,1))


#select row(s) with nan and remove it
df_1.isna().sum()
df_1[df_1.isna().any(axis=1)]
df_1.dropna(inplace=True)


#turn -99 with median becuz data is heavily skewed
plt.hist(df_1['perf_6_month_avg'][df_1['perf_6_month_avg']!=-99])
df_1['perf_6_month_avg'][df_1['perf_6_month_avg']!=-99].mean()
df_1['perf_6_month_avg'][df_1['perf_6_month_avg']!=-99].median()

x = df_1['perf_6_month_avg']==-99
x.sum()/len(df_1['perf_6_month_avg'])

df_1['perf_6_month_avg'][df_1['perf_6_month_avg']==-99]= df_1['perf_6_month_avg'][df_1['perf_6_month_avg']!=-99].median()


plt.hist(df_1['perf_12_month_avg'][df_1['perf_12_month_avg']!=-99])
df_1['perf_12_month_avg'][df_1['perf_12_month_avg']!=-99].mean()
df_1['perf_12_month_avg'][df_1['perf_12_month_avg']!=-99].median()

x = df_1['perf_12_month_avg']==-99
x.sum()/len(df_1['perf_12_month_avg'])

df_1['perf_12_month_avg'][df_1['perf_12_month_avg']==-99]= df_1['perf_12_month_avg'][df_1['perf_12_month_avg']!=-99].median()



#check unique sku
df_1.nunique()
df_2.nunique()

#check data type
df_1.info()
df_2.info()


##data transform

#transform sku into text
df_1['sku']= df_1['sku'].astype(str)

df_2['SKU']= df_2['SKU'].astype(str)


#Label Encoder
le = LabelEncoder()

df_1[['potential_issue','deck_risk','oe_constraint','ppap_risk','stop_auto_buy','rev_stop','went_on_backorder']]= df_1[['potential_issue','deck_risk','oe_constraint','ppap_risk','stop_auto_buy','rev_stop','went_on_backorder']].apply(le.fit_transform)




#normalize to check outlier
min_max_scaler = MinMaxScaler()
df_1_norm = min_max_scaler.fit_transform(df_1.iloc[:,1:22])
df_1_norm = pd.DataFrame(df_1_norm,columns=df_1.iloc[:,1:22].columns)

df_2_norm = min_max_scaler.fit_transform(df_2.iloc[:,1:-1])
df_2_norm = pd.DataFrame(df_2_norm,columns=df_2.iloc[:,1:-1].columns)

#check outlier with boxplot
plt.subplot(1,2,1)
plt.title(label = 'BOP_normalized (after cleaning)')
plt.boxplot(df_1_norm)
x = range(1,22)
labels = df_1_norm.columns
plt.xticks(x,labels, rotation = 'vertical' )

plt.subplot(1,2,2)
plt.title(label = 'bopredict_normalized')
plt.boxplot(df_2_norm)
x = range(1,22)
labels = df_2_norm.columns
plt.xticks(x,labels, rotation = 'vertical')


#Calculate average deviation
df_1_norm.mad(axis=0)
df_2_norm.mad(axis=0)


#corr for integer and float
fig1 = plt.figure(figsize=(8,8)) 
x = df_1.iloc[:,[12,17,18,19,20,21]]
x=x.columns
sns.heatmap(df_1.loc[:, ~df_1.columns.isin(x)].corr(method='kendall'),vmin=-1,vmax=1,cmap=plt.cm.Blues,annot=True,fmt='.2f').set(title='Correlation Matrix Amongst Attributes and Predicted Variable of BOP')

fig2 = plt.figure(figsize=(8,8)) 
x = df_2.iloc[:,[12,17,18,19,20,21]]
x=x.columns
sns.heatmap(df_2.loc[:, ~df_2.columns.isin(x)].corr(method='kendall'),vmin=-1,vmax=1,cmap=plt.cm.Blues,annot=True,fmt='.2f').set(title='Correlation Matrix Amongst Attributes and Predicted Variable of bopredict')

#Chi square BOP
x = df_1.iloc[:,[12,17,18,19,20,21,22]]
column_names=x.columns

chisqmatrix= pd.DataFrame(x,columns=column_names,index=column_names)

outercnt=0
innercnt=0
for icol in x.columns:
    
    for jcol in column_names:
        
       mycrosstab=pd.crosstab(x[icol],x[jcol])
       #print (mycrosstab)
       stat,p,dof,expected=stats.chi2_contingency(mycrosstab)
       chisqmatrix.iloc[outercnt,innercnt]=round(p,3)
       cntexpected=expected[expected<5].size
       perexpected=((expected.size-cntexpected)/expected.size)*100
      
       #print (icol)
       #print (jcol)
       if perexpected<20:
            chisqmatrix.iloc[outercnt,innercnt]=2
       #print (perexpected) 
       if icol==jcol:
           chisqmatrix.iloc[outercnt,innercnt]=0.00
       #print (expected) 
       innercnt=innercnt+1
    #print (outercnt) 
    outercnt=outercnt+1
    innercnt=0
    
sns.heatmap(chisqmatrix.astype(np.float64),vmin=-1,vmax=1,cmap=plt.cm.Blues,annot=True,fmt='.2f').set(title='Chi Square test for BOP')

#Chi square bopredict
x = df_2.iloc[:,[12,17,18,19,20,21,22]]
column_names=x.columns

chisqmatrix= pd.DataFrame(x,columns=column_names,index=column_names)

outercnt=0
innercnt=0
for icol in x.columns:
    
    for jcol in column_names:
        
       mycrosstab=pd.crosstab(x[icol],x[jcol])
       #print (mycrosstab)
       stat,p,dof,expected=stats.chi2_contingency(mycrosstab)
       chisqmatrix.iloc[outercnt,innercnt]=round(p,3)
       cntexpected=expected[expected<5].size
       perexpected=((expected.size-cntexpected)/expected.size)*100
      
       #print (icol)
       #print (jcol)
       if perexpected<20:
            chisqmatrix.iloc[outercnt,innercnt]=2
       #print (perexpected) 
       if icol==jcol:
           chisqmatrix.iloc[outercnt,innercnt]=0.00
       #print (expected) 
       innercnt=innercnt+1
    #print (outercnt) 
    outercnt=outercnt+1
    innercnt=0
    
sns.heatmap(chisqmatrix.astype(np.float64),vmin=-1,vmax=1,cmap=plt.cm.Blues,annot=True,fmt='.2f').set(title='Chi Square test for bopredict')


## Modeling (With Under Sampling)

# Initialize the StratifiedKFold for k-fold cross-validation
test_size = 0.2
sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)

# Initialize the RandomUnderSampler
under_sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X = df_1.drop(columns = ['went_on_backorder','sku'])
y = df_1['went_on_backorder']


# Initialize and train the XGBoost classifier
xgb_model = XGBClassifier(random_state=42)

# Define the parameter grid for RandomizedSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}


# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(xgb_model, param_distributions=param_grid, n_iter=10, scoring='roc_auc', cv=None, random_state=42)

#With undersampling
for train_index, test_index in sss.split(X, y):
    with_sampling_X_train, with_sampling_X_test = X.iloc[train_index], X.iloc[test_index]
    with_sampling_y_train, with_sampling_y_test = y.iloc[train_index], y.iloc[test_index]

# Apply undersampling to the training data
X_train_resampled, y_train_resampled = under_sampler.fit_resample(with_sampling_X_train,with_sampling_y_train)

with_sampling_rand_search = random_search.fit(X_train_resampled,y_train_resampled) 
with_sampling_best_model = with_sampling_rand_search.best_estimator_
    
# Make predictions on the test data
with_sampling_y_pred = with_sampling_best_model.predict(with_sampling_X_test)
    
# Calculate metrics and store them in the lists
with_sampling_accuracy = accuracy_score(with_sampling_y_test,with_sampling_y_pred)
with_sampling_sensitivity = recall_score(with_sampling_y_test,with_sampling_y_pred)
with_sampling_specificity = recall_score(with_sampling_y_test,with_sampling_y_pred, pos_label=0)
with_sampling_precision = precision_score(with_sampling_y_test,with_sampling_y_pred)
with_sampling_f1 = f1_score(with_sampling_y_test,with_sampling_y_pred)
with_sampling_roc_auc = roc_auc_score(with_sampling_y_test,with_sampling_y_pred)
with_sampling_best_parameters = with_sampling_rand_search.best_params_
with_sampling_confusion_matrix = confusion_matrix(with_sampling_y_test,with_sampling_y_pred)

# Calculate SHAP values for the test data
shap_explainer = shap.Explainer(with_sampling_best_model)
shap_values = shap_explainer(with_sampling_X_test)
type(shap_values)

# Plot a SHAP summary plot (replace with your preferred SHAP visualization)
shap.summary_plot(shap_values, with_sampling_X_test, show=False)
plt.title("SHAP Summary Plot (Under Sampling)")
plt.show()

# Calculate the metric scores across all folds
print("Accuracy:", with_sampling_accuracy)
print("Precision:", with_sampling_precision)
print("Sensitivity:", with_sampling_sensitivity)
print("Specificity:", with_sampling_specificity)
print("F1-score:", with_sampling_f1)
print("AUC-ROC:", with_sampling_roc_auc)
print("confusion matrix:", with_sampling_confusion_matrix)
print("Best Parameters:", with_sampling_best_parameters)



## Modeling (Without undersampling)

# Initialize the StratifiedKFold for k-fold cross-validation
test_size = 0.2
sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)

# Initialize the RandomUnderSampler
under_sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X = df_1.drop(columns = ['went_on_backorder','sku'])
y = df_1['went_on_backorder']


# Initialize and train the XGBoost classifier
xgb_model = XGBClassifier(random_state=42)

# Define the parameter grid for RandomizedSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(xgb_model, param_distributions=param_grid, n_iter=10, scoring='roc_auc', cv=None, random_state=42)

for train_index, test_index in sss.split(X, y):
    X_train_without_sampling, X_test_without_sampling = X.iloc[train_index], X.iloc[test_index]
    y_train_without_sampling, y_test_without_sampling = y.iloc[train_index], y.iloc[test_index]
    
random_search.fit(X_train_without_sampling,y_train_without_sampling) 
without_sampling_best_model = random_search.best_estimator_
    
# Make predictions on the test data
y_pred_without_sampling = without_sampling_best_model.predict(X_test_without_sampling)
    
# Calculate metrics and store them in the lists
without_sampling_accuracy = accuracy_score(y_test_without_sampling, y_pred_without_sampling)
without_sampling_sensitivity = recall_score(y_test_without_sampling, y_pred_without_sampling)
without_sampling_specificity = recall_score(y_test_without_sampling, y_pred_without_sampling, pos_label=0)
without_sampling_precision = precision_score(y_test_without_sampling, y_pred_without_sampling)
without_sampling_f1 = f1_score(y_test_without_sampling, y_pred_without_sampling)
without_sampling_roc_auc = roc_auc_score(y_test_without_sampling, y_pred_without_sampling)
without_sampling_best_parameters = random_search.best_params_
without_sampling_confusion_matrix = confusion_matrix(y_test_without_sampling, y_pred_without_sampling)
    
# Calculate SHAP values for the test data
shap_explainer = shap.Explainer(without_sampling_best_model)
shap_values = shap_explainer(X_test_without_sampling)

# Plot a SHAP summary plot (replace with your preferred SHAP visualization)
shap.summary_plot(shap_values, X_test_without_sampling, show=False)
plt.title("SHAP Summary Plot (Without Under Sampling)")
plt.show()

# Calculate the metric scores across all folds
print("Accuracy:", without_sampling_accuracy)
print("Precision:", without_sampling_precision)
print("Sensitivity:", without_sampling_sensitivity)
print("Specificity:", without_sampling_specificity)
print("F1-score:", without_sampling_f1)
print("AUC-ROC:", without_sampling_roc_auc)
print("confusion matrix:", without_sampling_confusion_matrix)
print("Best Parameters:", without_sampling_best_parameters)



## Modeling (With SMOTE)

# Initialize the StratifiedKFold for k-fold cross-validation
test_size = 0.2
sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)

# Initialize the RandomUnderSampler
smote = SMOTE(sampling_strategy='minority',random_state=42)
X = df_1.drop(columns = ['went_on_backorder','sku'])
y = df_1['went_on_backorder']

# Initialize and train the XGBoost classifier
xgb_model = XGBClassifier(random_state=42)

# Define the parameter grid for RandomizedSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}


# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(xgb_model, param_distributions=param_grid, n_iter=10, scoring='roc_auc', cv=None, random_state=42)

#With undersampling
for train_index, test_index in sss.split(X, y):
    X_train_SMOTE, X_test_SMOTE = X.iloc[train_index], X.iloc[test_index]
    y_train_SMOTE, y_test_SMOTE = y.iloc[train_index], y.iloc[test_index]
    
# Apply SMOTE to the training data
X_train_SMOTE_1, y_train_SMOTE_1 = smote.fit_resample(X_train_SMOTE,y_train_SMOTE)

rand_search_SMOTE = random_search.fit(X_train_SMOTE_1,y_train_SMOTE_1) 
best_model_SMOTE = rand_search_SMOTE.best_estimator_


# Make predictions on the test data
y_pred_SMOTE_1 = best_model_SMOTE.predict(X_test_SMOTE)

# Calculate metrics and store them in the lists
SMOTE_accuracy = accuracy_score(y_test_SMOTE,y_pred_SMOTE_1)
SMOTE_sensitivity = recall_score(y_test_SMOTE,y_pred_SMOTE_1)
SMOTE_specificity = recall_score(y_test_SMOTE,y_pred_SMOTE_1, pos_label=0)
SMOTE_precision = precision_score(y_test_SMOTE,y_pred_SMOTE_1)
SMOTE_f1 = f1_score(y_test_SMOTE,y_pred_SMOTE_1)
SMOTE_roc_auc = roc_auc_score(y_test_SMOTE,y_pred_SMOTE_1)
SMOTE_best_parameters = rand_search_SMOTE.best_params_
SMOTE_confusion_matrix = confusion_matrix(y_test_SMOTE,y_pred_SMOTE_1)

# Calculate SHAP values for the test data
shap_explainer = shap.Explainer(best_model_SMOTE)
test_shap_values = shap_explainer(X_test_SMOTE)

# Plot a SHAP summary plot (replace with your preferred SHAP visualization)
shap.summary_plot(shap_values, X_test_SMOTE, show=False)
plt.title("SHAP Summary Plot (SMOTE)")
plt.show()

# Calculate the metric scores across all folds
print("Accuracy:", SMOTE_accuracy)
print("Precision:", SMOTE_precision)
print("Sensitivity:", SMOTE_sensitivity)
print("Specificity:", SMOTE_specificity)
print("F1-score:", SMOTE_f1)
print("AUC-ROC:", SMOTE_roc_auc)
print("confusion matrix:", SMOTE_confusion_matrix)
print("Best Parameters:", SMOTE_best_parameters)


## Modeling (SMOTE, RF)

# Initialize the StratifiedKFold for k-fold cross-validation
test_size = 0.2
sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)

# Initialize the RandomUnderSampler
rus = RandomUnderSampler(sampling_strategy='auto',random_state=42)
X = df_1.drop(columns = ['went_on_backorder','sku'])
y = df_1['went_on_backorder']

for train_index, test_index in sss.split(X, y):
    X_train_rf_RUS, X_test_rf_RUS = X.iloc[train_index], X.iloc[test_index]
    y_train_rf_RUS, y_test_rf_RUS = y.iloc[train_index], y.iloc[test_index]
    

# Initialize the RandomForest classifier
rf_model = RandomForestClassifier(random_state=42)
X_train_resampled, y_train_resampled = rus.fit_resample(X_train_rf_RUS, y_train_rf_RUS)

# Define the hyperparameter grid for random search
param_dist = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2', None]
}

# Initialize the RandomForest classifier
rf_model = RandomForestClassifier(random_state=42)

# Initialize and perform random search for hyperparameter tuning
random_search = RandomizedSearchCV(rf_model, param_distributions=param_dist, n_iter=10, scoring='roc_auc', cv=3,verbose=2, random_state=42, n_jobs=-1)
random_search.fit(X_train_resampled, y_train_resampled)

# Get the best model from random search
best_model = random_search.best_estimator_

# Make predictions on the test data
y_pred_rf_RUS = best_model.predict(X_test_rf_RUS)

# Calculate metrics
rf_RUS_accuracy = accuracy_score(y_test_rf_RUS, y_pred_rf_RUS)
rf_RUS_sensitivity = recall_score(y_test_rf_RUS, y_pred_rf_RUS)
rf_RUS_specificity = recall_score(y_test_rf_RUS, y_pred_rf_RUS, pos_label=0)
rf_RUS_precision = precision_score(y_test_rf_RUS, y_pred_rf_RUS)
rf_RUS_f1 = f1_score(y_test_rf_RUS, y_pred_rf_RUS)
rf_RUS_roc_auc = roc_auc_score(y_test_rf_RUS, y_pred_rf_RUS)
rf_RUS_confusion_matrix = confusion_matrix(y_test_rf_RUS, y_pred_rf_RUS)
rf_RUS_best_parameters = random_search.best_params_


# Print best hyperparameters and metrics
print("Accuracy:", rf_RUS_accuracy)
print("Precision:", rf_RUS_precision)
print("Sensitivity:", rf_RUS_sensitivity)
print("Specificity:", rf_RUS_specificity)
print("F1-score:", rf_RUS_f1)
print("AUC-ROC:", rf_RUS_roc_auc)
print("confusion matrix:", rf_RUS_confusion_matrix)
print("Best Hyperparameters:", rf_RUS_best_parameters)


# Calculate SHAP values for the test data
RF_shap_explainer = shap.TreeExplainer(best_model)
RF_shap_values = RF_shap_explainer(X_test_rf_RUS)


reshaped_values = RF_shap_values.values[:, :, 0].reshape((385987, 21))


# Plot a SHAP summary plot (replace with your preferred SHAP visualization)
shap.summary_plot(reshaped_values, X_test_rf_RUS)
plt.title("SHAP Summary Plot (RUS)")
plt.show()

shap.summary_plot(reshaped_values, X_test_rf_RUS, plot_type='bar')

RF_shap_values

X_test_rf_RUS.shape
X_train_rf_RUS.shape

np.random.seed(42)

# Generate synthetic data
num_samples = 100
num_features = 5

X= np.random.rand(num_samples, num_features)  # Random features between 0 and 1
y = np.random.choice([0, 1], size=num_samples)  # Random binary labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Initialize the SHAP explainer
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values[1], X_test)

