import sys,os
import numpy as np
import scipy as scipy
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import shap
import urllib
#import requests
import zipfile
#import seaborn
from io import StringIO
import warnings
warnings.filterwarnings('ignore')
from scipy import sparse
from pandas.plotting import scatter_matrix
from datetime import datetime
from pprint import pprint
from sklearn import tree

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import classification_report, roc_auc_score
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, make_scorer, roc_curve, auc, f1_score, accuracy_score
from sklearn.metrics import precision_recall_curve
#from sklearn.externals import joblib
import pickle
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix


################################################################
#Read and parse the dataset
################################################################
print("\nReading in the input data...")

# Prompt the user for the input file path
#file_path = input("Enter the path to the input CSV file: ")

# Read the CSV file using the provided input file path
df = pd.read_csv(train_test_all.csv)

#df = pd.read_csv('/home/51t/models/Training_Testing_datasets/n_estimators_100/scaled/SVM/rbf/combined_train_test.csv')
var_columns = [c for c in df.columns if c not in('CDR3.beta', 'antigen_epitope','mhc.a','label','negative.source','license')]
# Nylonase,Class
X = df.loc[:, var_columns]
y = df.loc[:, 'label']


################################################################
#Rescale data and split into training and testing sets
################################################################
print("\nRescaling data...") #Dividing data into training and test sets..."


#NOTE: we use X_scaled and y throughout. If you want to divide data into training and test data sets, you can use the following
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1234)

correlation_matrix = X_train.corr()

def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

corr_features = correlation(X_train, 0.8)
#print(len(set(corr_features)))

corr_features = list(corr_features)

print("Removing correlated features ")

X_train = X_train.drop(corr_features,axis=1)
X_test = X_test.drop(corr_features,axis=1)

print("Scaling the features ")

# Scale the features
X_train_scaled = preprocessing.scale(X_train)
X_test_scaled = preprocessing.scale(X_test)


################################################################
#Begin Parameter Grid Search with Cross-Validation 
################################################################
#print("\nPerforming Parameter Search for Rbf-based Model...")

roc_scorer = make_scorer(roc_auc_score) #note, can use this or 'f1' for scoring below
#acc_scorer = make_scorer(accuracy_score)

print("Training... ")

#Uncomment below two lines if you chose to optimize hyperparameters for your training data and comment out default parameters section
#parameters= {'C': C_range, 'gamma': gamma_range, 'kernel':['rbf'], 'class_weight':['balanced']}
#classifier=GridSearchCV(SVC(), parameters, cv=10, scoring=acc_scorer, n_jobs=-1)

#Default parameters for SVM model
parameters= {'C': [1.0], 'gamma': ['scale'], 'kernel':['rbf'], 'class_weight':['balanced']}
#classifier=GridSearchCV(SVC(), parameters, cv=None, scoring=roc_scorer, n_jobs=-1)
classifier=GridSearchCV(SVC(probability=True), parameters, cv=None, scoring=roc_scorer, n_jobs=-1)

#fit the model
clf=classifier.fit(X_train_scaled, y_train)
print("\n\tThe best CV parameters for rbf model are [" + str(classifier.best_params_) + "] with a score on train data of [" + str(classifier.best_score_) + "]")


#predictions=clf.predict(X_test_scaled)
#calculate_metrics(y_test, predictions)

print("Training Done ")

print("Evaluating...")
y_pred=clf.predict(X_test_scaled)
print("\nScore on test data using rbf model are AUC of ROC = {}".format(roc_auc_score(y_test,y_pred)))


#y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Calculate specificity
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)
print("TN:",tn, "FP:",fp, "FN:",fn, "TP:",tp)

# Print the results
print("Results")
print("Accuracy: {:.2f}".format(accuracy))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))
print("Specificity: {:.2f}".format(specificity))

#SHAP analysis
# Get feature names
feature_names = X_train.columns
 
X_importance=X_train_scaled
 
# Create the KernelExplainer
explainer = shap.KernelExplainer(model=classifier.predict_proba, data=X_importance, link='logit')
 
shap_values = explainer.shap_values(X_test_scaled)
shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names, max_display=50, plot_size=[8,12])
 
