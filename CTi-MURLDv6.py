# -*- coding: utf-8 -*-
"""
*****************************************************************************************************************************************
Created on Mon Aug 16 02:08:31 2021
@author: Fuad A. Ghaleb(fuadeng@gmail.com, abdulgaleel@utm.my)
This code is the implmentation of the CTI-MURLD model to detcted malicious URLs
Three Types of features are included:
    1- URL based features
    2- CyberThreats Intelligence collected from google search engine
    3- Whois Information
    
CTI-MURLD model combines 3 RF based predectors for predetection and ANN based classifier for final decison 
Each RF predictor constructed in 6 phases: 
    1- data collection, 
    2- features preprocessing,
    3- features extraction, 
    4- features representation, 
    5- features selection, 
    6- training of the ensemble learning based RF prediction, 
For the final decision making, the probablistic outputs of the RF predictores are fed to ANN based classfier.
The ANN based classfier is trained using the probablistic outputs of the RF predictores.
******************************************************************************************************************************************

"""

#import whois
import pandas as pd
import numpy as np
import time

from urllib.parse import urlparse
from nltk.corpus import stopwords
import string

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report


# %%
"""
******************************************************************************************************************************************
intilization
******************************************************************************************************************************************
"""
# %%

#Results Lists
#true negative
tn_list=[]
#false positive
fp_list=[]
#false negative
fn_list=[]
#true positive
tp_list=[]

#RF classfiers accuracy 
classifiers_list =[]

# data fram to store the propaplistic outputs of the classfiers 
df_class_output = pd.DataFrame()

#max number of features selected
f=5000

# %%
"""
*****************************************************************************************************************************************
load the datasets
******************************************************************************************************************************************
"""
# %%
# fix random seed for reproducibility
np.random.seed(7)
ds = pd.read_csv("E:/Malicious Website Detection/url_whois_list.csv")

#Number of samples 
n= ds.shape[0]

#Sample id
j=0
whois_list=[]

#ds.to_csv("url_whois_list.csv")
google_list=[]
label_list=[]
url_list=[]
""" Loading The CTI-google and Whois Data """
for i in range(j, n):
          
    #print(result)
    j=j+1
    #print(j)
    url_list.append(ds.url.values[i])
    label_list.append(ds.label.values[i])
    try:
        text_file = open("E:/Malicious Website Detection/whois/" + str(i) + ".txt", "r", encoding="utf-8")
        resultg = ds.url.values[i] + " " + text_file.read()
        text_file.close()
    except:
        result =ds.url.values[i] + " " + "empty"     
    try:
        text_file = open("E:/Malicious Website Detection/gCTI/" + str(i) + "g.txt", "r", encoding="utf-8")
        resultw = ds.url.values[i] + " " + text_file.read()
        text_file.close()
    except:
        result = ds.url.values[i] + " " + "empty"
    whois_list.append(resultw)
    google_list.append(resultg)

df = pd.DataFrame()

df['google_cti']=google_list
df['whois_info']=whois_list
df['url']=url_list
df['label']=label_list
df['label'] = df['label'].replace(['good'],'0')
df['label'] = df['label'].replace(['bad'],'1')


# %%
"""
******************************************************************************************************************************************
Data Preprocessing 
******************************************************************************************************************************************
"""
# %%

#convert to lower 
df.loc[:,"whois_info"] = df.whois_info.apply(lambda x : str.lower(x))
df.loc[:,"google_cti"] = df.google_cti.apply(lambda x : str.lower(x))

import re
df.loc[:,"whois_info"] = df.whois_info.apply(lambda x : " ".join(re.findall('[\w]+',x)))
df.loc[:,"google_cti"] = df.google_cti.apply(lambda x : " ".join(re.findall('[\w]+',x)))


# Import stopwords with nltk.
from nltk.corpus import stopwords
stop = stopwords.words('english')

from stop_words import get_stop_words
stop_words = get_stop_words('en')

def remove_stopWords(s):
    '''For removing stop words
    '''
    s = ' '.join(word for word in s.split() if word not in stop_words)
    return s


#Remove Stop Word
df.loc[:,"whois_info"] = df.whois_info.apply(lambda x: remove_stopWords(x))
ds['label'] = ds['label'].replace(['good'],'0')
ds['label'] = ds['label'].replace(['bad'],'1')
#Save a copy of the preprocessed data to the hardisk
df.to_csv("url_cti_whois_list.csv")

#read a copy of the preprocessed data to the hardisk
df = pd.read_csv("url_cti_whois_list.csv")

# %%
"""
******************************************************************************************************************************************
1--Train and Test the URL Predictor using RF algorithm (URL-RF)
******************************************************************************************************************************************
"""
# %%


# %%
""" URL Features Exteraction 
"""
# %%

#Training Set 80%
df_train = df.sample(frac=0.7)

#Testing Set
df_test = df.drop(df_train.index)

x_train = df_train.url
y_train=df_train.label.values

x_test = df_test.url
y_test = df_test.label.values


# %%
""" URL Features Representation  
"""
# %%
# characters level tf-idf
tfidf_vect_ngram_url = TfidfVectorizer(analyzer='char', ngram_range=(2,3))
tfidf_vect_ngram_url.fit(x_train)
xtrain_tfidf_ngram_url =  tfidf_vect_ngram_url.transform(x_train) 
xvalid_tfidf_ngram_url =  tfidf_vect_ngram_url.transform(x_test) 


# %%
""" URL Features Selection  
"""
# %%
#from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.feature_selection import SelectKBest, SelectPercentile, mutual_info_classif
#mi_score = MIC(xtrain_tfidf_ngram_chars,y_train)
#print(mi_score)

#Number of selected features  (all features in the URL case) 
selector = SelectKBest(mutual_info_classif, k=f)
X_reduced_training = selector.fit_transform(xtrain_tfidf_ngram_url, y_train)
X_reduced_training.shape
X_reduced_valid = selector.transform(xvalid_tfidf_ngram_url);



# %%
""" URL RF Training and Predection  -- get the probablistic of predicting zero and one
"""
# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
#Create a Gaussian Classifier
clf_url = RandomForestClassifier(n_estimators=100).fit(X_reduced_training, y_train)

#testing the URL-RF classifier
predicted1 = clf_url.predict(X_reduced_training)
predicted2 = clf_url.predict(X_reduced_valid)

print(np.mean(predicted2 == y_test) )
tn, fp, fn, tp = confusion_matrix(predicted2,  y_test).ravel()

print("URL Accuracy:" , (tp+tn)/(tn+tp+fp+fn))

# Store Testing Results of the url classfier
classifiers_list.append("RF-url")
tn_list.append(tn)
fp_list.append(fp)
fn_list.append(fn)
tp_list.append(tp)
df_class_output['RF-url'] = list(map(int, predicted1)) + list(map(int, predicted2))


#get the accumlated probablistic outputs of  predicting 1 and predecting 0
url_Rf_predicted_output1 = clf_url.predict_proba(X_reduced_training)
url_Rf_predicted_output2 = clf_url.predict_proba(X_reduced_valid)

#store into the dataset for final classfication
df_class_output['RF-url_pred0'] = list(url_Rf_predicted_output1[:,0]) + list(url_Rf_predicted_output2[:,0]) 
df_class_output['RF-url_pred1'] = list(url_Rf_predicted_output1[:,1]) + list(url_Rf_predicted_output2[:,1]) 




# %%
"""
******************************************************************************************************************************************
2--Train and Test the Whois Predictor Using RF algorithm (whois-RF)
******************************************************************************************************************************************
"""
# %%

# %%
"""
 Whois features pereperation 
"""
# %%

x_train = df_train.whois_info
y_train=df_train.label.values

x_test = df_test.whois_info
y_test = df_test.label.values

# %%
""" whois Features Exteraction and representation using Ngram and TFIDF
"""
# %%
# characters level tf-idf
tfidf_vect_ngram_whois = TfidfVectorizer(analyzer='word', ngram_range=(1,3))
tfidf_vect_ngram_whois.fit(x_train)
xtrain_tfidf_ngram_whois =  tfidf_vect_ngram_whois.transform(x_train) 
xvalid_tfidf_ngram_whois =  tfidf_vect_ngram_whois.transform(x_test) 


# %%
""" whois Features Selection top 5000 (f) 
"""
# %%
X_reduced_training = selector.fit_transform(xtrain_tfidf_ngram_whois, y_train)
X_reduced_training.shape
X_reduced_valid = selector.transform(xvalid_tfidf_ngram_whois);


# %%
""" whois RF Training and Predection  -- get the probablistic of predicting zero and one
"""
# %%
#Create a RF Predictor (whois)
clf_whois = RandomForestClassifier(n_estimators=100).fit(X_reduced_training, y_train)

#Test  the Classfication Accuracy whois-RF
predicted1 = clf_whois.predict(X_reduced_training)
predicted2 = clf_whois.predict(X_reduced_valid)

print(np.mean(predicted2 == y_test) )
tn, fp, fn, tp = confusion_matrix(predicted2,  y_test).ravel()

print("Whois-CTi Accuracy:" , (tp+tn)/(tn+tp+fp+fn))


classifiers_list.append("RF-whois")
tn_list.append(tn)
fp_list.append(fp)
fn_list.append(fn)
tp_list.append(tp)
df_class_output['RF-whois'] = list(map(int, predicted1)) + list(map(int, predicted2))


#get the probablistic outputs of  predicting 1 and predecting 0
whois_Rf_predicted_output1 = clf_whois.predict_proba(X_reduced_training)
whois_Rf_predicted_output2 = clf_whois.predict_proba(X_reduced_valid)

#store into the dataset for final classfication
df_class_output['RF-whois_pred0'] = list(whois_Rf_predicted_output1[:,0]) + list(whois_Rf_predicted_output2[:,0] )
df_class_output['RF-whois_pred1'] = list(whois_Rf_predicted_output1[:,1]) + list(whois_Rf_predicted_output2[:,1] )



# %%
"""
******************************************************************************************************************************************
3--Train and Test the CTI google Predictor Using RF algorithm (googleCTi_RF)
******************************************************************************************************************************************
"""
# %%


# %%
""" google training and testing samples pereperation 
"""
# %%

x_test = df_test.google_cti
y_train=df_train.label.values

x_train = df_train.google_cti
y_test = df_test.label.values

# %%
""" googlecti Features Exteraction and representation using Ngram and TFIDF
"""
# %%
# words level tf-idf
tfidf_vect_ngram_googlecti = TfidfVectorizer(analyzer='word', ngram_range=(1,3))
tfidf_vect_ngram_googlecti.fit(x_train)
xtrain_tfidf_ngram_googlecti =  tfidf_vect_ngram_googlecti.transform(x_train) 
xvalid_tfidf_ngram_googlecti =  tfidf_vect_ngram_googlecti.transform(x_test) 


# %%
""" googlecti Features Selection top 5000 (f) 
"""
# %%
X_reduced_training = selector.fit_transform(xtrain_tfidf_ngram_googlecti, y_train)
X_reduced_training.shape
X_reduced_valid = selector.transform(xvalid_tfidf_ngram_googlecti);


# %%
""" googlecti RF Training and Predection  -- get the probablistic of predicting zero and one
"""
# %%
#Train the googleCTi_RF
clf_google_RF = RandomForestClassifier(n_estimators=100).fit(X_reduced_training, y_train)

#Test the googleCTi_RF
predicted1 = clf_google_RF.predict(X_reduced_training)
predicted2 = clf_google_RF.predict(X_reduced_valid)

print(np.mean(predicted2 == y_test) )
tn, fp, fn, tp = confusion_matrix(predicted2,  y_test).ravel()

print("Google-CTi Accuracy:" , (tp+tn)/(tn+tp+fp+fn))

classifiers_list.append("RF-google")
tn_list.append(tn)
fp_list.append(fp)
fn_list.append(fn)
tp_list.append(tp)
df_class_output['RF-google'] = list(map(int, predicted1)) + list(map(int, predicted2))


#get the probablistic outputs of the predicting 1 and predecting 0
googleCTi_RF_predicted_outputs1 = clf_google_RF.predict_proba(X_reduced_training)
googleCTi_RF_predicted_outputs2 = clf_google_RF.predict_proba(X_reduced_valid)

#store into the dataset for final classfication
df_class_output['RF-google_pred0'] = list(googleCTi_RF_predicted_outputs1[:,0]) + list(googleCTi_RF_predicted_outputs2[:,0]) 
df_class_output['RF-google_pred1'] = list(googleCTi_RF_predicted_outputs1[:,1]) + list(googleCTi_RF_predicted_outputs2[:,1]) 

df_class_output['label'] = list(y_train)  + list(y_test)


# %%
"""  
******************************************************************************************************************************************
Training and Testing the  ANN based Decision Making
******************************************************************************************************************************************
"""
# %%

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

"""
in df_class_output dataset
    features 1 and 2 are the output of the URL-RF
    features 4 and 5 are the output of the whois-RF
    features 7 and 8 are the output of the googleCTi-RF
    feature 9 is the true lable (the ground truth)
"""
 
X= df_class_output.iloc[:,[1,2,4,5,7,8]]
y= df_class_output.iloc[:,9]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,test_size=0.30, random_state=1)

#Train the MLP for final decison making
clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)

#get the final decison making about the class 
predicted = clf.predict_proba(X_test)
clf.score(X_test, y_test)
tn, fp, fn, tp = confusion_matrix(predicted,  y_test).ravel()

print("CTI MURLD Accuracy:" , (tp+tn)/(tn+tp+fp+fn))
