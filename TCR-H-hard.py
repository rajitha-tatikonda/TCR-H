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
import pandas as pd

################################################################
# Routines to compute performance metrics
################################################################
def calculate_metrics(true_vals,predicted_vals):
    if len(true_vals) !=len(predicted_vals):
        print("Error! Lengths of arrays do not match")
        return 0
    TP=0 #True positives
    FP=0 #False positives
    TN=0 #True negatives
    FN=0 #False negatives

    for i in range(0,len(true_vals)):
        if (true_vals[i]==0 and predicted_vals[i]==0):
            TN+=1
        if (true_vals[i]==0 and predicted_vals[i]==1):
            FP+=1
        if (true_vals[i]==1 and predicted_vals[i]==0):
            FN+=1
        if (true_vals[i]==1 and predicted_vals[i]==1):
            TP+=1

    print("\tTP=",TP," TN=",TN," FP=",FP," FN=",FN)
    print("\ttotal=", TP+FP+TN+FN)
    accuracy= float((TP+TN))/float(TP+TN+FP+FN)
    precision=0.0
    if (TP+FP)>0:
        precision= float((TP))/float(TP+FP)
    recall=0.0
    if (TP+FN)>0:
        recall= float((TP))/float(TP+FN)
    specificity=0.0
    if (TN+FP)>0:
        specificity= float((TN))/float(TN+FP)
    f_score=0.0
    if (precision+recall)>0.0:
        f_score=2.0*(precision*recall)/(precision+recall)

    print("\taccuracy=", accuracy, "  precision=", precision, "  recall= ",recall, "  specificity= ",specificity, " F-score=", f_score)
    return


################################################################
#Read and parse the dataset
################################################################
print("\nReading in the training data...")

# Load the training and test data
df_test = pd.read_csv('test_data.csv')

df_train = pd.read_csv('train_data.csv')
var_columns = [c for c in df_train.columns if c not in('CDR3.beta', 'antigen_epitope','mhc.a','label','negative.source','license')]
# Nylonase,Class
X_train = df_train.loc[:, var_columns]
y_train = df_train.loc[:, 'label']

X_test = df_test.loc[:, var_columns]
y_test = df_test.loc[:, 'label']

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
print(len(set(corr_features)))

corr_features = list(corr_features)

print("Removing correlated features ")
X_train = X_train.drop(corr_features,axis=1)
X_test = X_test.drop(corr_features,axis=1)

# Split the data into feature and target variables
y_train = df_train['label']
y_test = df_test['label']



################################################################
#Rescale data and split into training and testing sets
################################################################
print("\nRescaling data...") #Dividing data into training and test sets..."
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import classification_report, roc_auc_score
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

print("Scaling the features ")
# Scale the features
X_train_scaled = preprocessing.scale(X_train)
X_test_scaled = preprocessing.scale(X_test)


#NOTE: we use X_scaled and y throughout. If you want to divide data into training and test data sets, you can use the following
#X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=1234)

from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

################################################################
#Begin Parameter Grid Search with Cross-Validation 
################################################################
print("\nPerforming Parameter Search for Rbf-based Model...")

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, make_scorer, roc_curve, auc, f1_score, accuracy_score,precision_score, recall_score, confusion_matrix
from sklearn.metrics import precision_recall_curve
#from sklearn.externals import joblib
import pickle
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')

roc_scorer = make_scorer(roc_auc_score) #note, can use this or 'f1' for scoring below
#acc_scorer = make_scorer(accuracy_score)

print("Training... ")
C_range = [0.01, 0.1, 1.0, 10.0, 100.0]
gamma_range = np.logspace(-4,2,5)

#Uncomment below two lines if you chose to optimize hyperparameters for your training data and comment out default parameters section
#parameters= {'C': C_range, 'gamma': gamma_range, 'kernel':['rbf'], 'class_weight':['balanced']}
#classifier=GridSearchCV(SVC(), parameters, cv=10, scoring=acc_scorer, n_jobs=-1)

#Default parameters for SVM model
parameters= {'C': [1.0], 'gamma': ['scale'], 'kernel':['rbf'], 'class_weight':['balanced']}
#classifier=GridSearchCV(SVC(), parameters, cv=None, scoring=acc_scorer, n_jobs=-1)
classifier=GridSearchCV(SVC(probability=True), parameters, cv=None, scoring=acc_scorer, n_jobs=-1)

#Fit the model

clf=classifier.fit(X_train_scaled, y_train)
print "\n\tThe best CV parameters for rbf model are [" + str(classifier.best_params_) + "] with a score on train data of [" + str(classifier.best_score_) + "]"
#pickle.dump(clf, open('svm.pkl', 'wb'))
print("Training Done ")

print("Evaluating...")

predictions=clf.predict(X_test_scaled)
print("\nScore on test data using rbf model are AUC of ROC = {}".format(roc_auc_score(y_test,predictions)))
calculate_metrics(y_test, predictions)


# Save the trained model to a file
pickle.dump(clf, open('svm_uncorrelated.pkl', 'wb'))
#joblib.dump(svm_model, 'svm_model.pkl')

##SHAP analysis
# Get feature names
feature_names = X_train.columns
 
X_importance=X_train_scaled
 
# Create the KernelExplainer
explainer = shap.KernelExplainer(model=classifier.predict_proba, data=X_importance, link='logit')
 
shap_values = explainer.shap_values(X_test_scaled)
shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names, max_display=50, plot_size=[8,12])
 
