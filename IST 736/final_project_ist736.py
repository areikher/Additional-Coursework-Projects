#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 21:22:46 2022

@author: alison
"""

import nltk
import pandas as pd
import sklearn
import re  
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
## For Stemming
from nltk.tokenize import sent_tokenize, word_tokenize
import os

from nltk.stem.porter import PorterStemmer
STEMMER=PorterStemmer()


# Use NLTK's PorterStemmer in a function
def MY_STEMMER(str_input):
    #words = re.sub(r"[']", "", str_input).lower().split()
    words = re.sub(r"[^A-Za-z\-]", " ", str_input).lower().split()
    words = [STEMMER.stem(word) for word in words]
    #print(words)
    return words

def my_punctuation_remover(str_input):
    new_string = str_input.translate(str.maketrans('', '', string.punctuation))
    #return_lst = [new_string.lower().split() for word in new_string]
    new_string = new_string.lower().split()
    return_lst = list(new_string)
    return return_lst

import string
import numpy as np

MyVect_STEM=CountVectorizer(input='filename',
                        analyzer = 'word',
                        stop_words='english',
                        ##stop_words=["and", "or", "but"],
                        #token_pattern='(?u)[a-zA-Z]+',
                        #token_pattern=pattern,
                        tokenizer=MY_STEMMER,
                        #strip_accents = 'unicode',
                        lowercase = True
                        )



MyVect_IFIDF=TfidfVectorizer(input='filename',
                        analyzer = 'word',
                        stop_words='english',
                        lowercase = True,
                        #binary=True
                        )

MyVect_IFIDF_STEM=TfidfVectorizer(input='filename',
                        analyzer = 'word',
                        stop_words='english',
                        tokenizer=MY_STEMMER,
                        #strip_accents = 'unicode',
                        lowercase = True,
                        #binary=True
                        )

MyVect_Bernoulli=CountVectorizer(input='filename',
                        analyzer = 'word',
                        stop_words='english',
                        #tokenizer=MY_STEMMER,
                        #strip_accents = 'unicode',
                        lowercase = True,
                        binary=True
                                )

MyVect=CountVectorizer(input='filename',
                        analyzer = 'word',
                        stop_words='english',
                        ##stop_words=["and", "or", "but"],
                        #token_pattern='(?u)[a-zA-Z]+',
                        #token_pattern=pattern,
                        tokenizer=my_punctuation_remover,
                        #strip_accents = 'unicode',
                        lowercase = True
                        )

FinalDF_STEM=pd.DataFrame()
FinalDF_TFIDF=pd.DataFrame()
FinalDF_TFIDF_STEM=pd.DataFrame()
FinalDF_Bernoulli=pd.DataFrame()
FinalDF=pd.DataFrame()

for name in ["Alexander Clean", "Hamilton Clean"]:
    print(name)
    path= name
    builder=name+"DF"    
    #print(builder)
    builderB=name+"DFB"
    
    FileList=[]
    for item in os.listdir(path):
        print(path+ "\\" + item)
        next=path+ "/" + item
        if next == 'Alexander Clean/.DS_Store':
            continue
        else:
            FileList.append(next)  
        #print("full list...")
        print(FileList)
       
        ## Do for all three
        ## MyVect_STEM  and MyVect_IFIDF and MyVect_IFIDF_STEM
        X1=MyVect_STEM.fit_transform(FileList)
        X2=MyVect_IFIDF.fit_transform(FileList)
        X3=MyVect_IFIDF_STEM.fit_transform(FileList)
        X4=MyVect_Bernoulli.fit_transform(FileList)
        X5=MyVect.fit_transform(FileList)
       
       
        ColumnNames1=MyVect_STEM.get_feature_names()
        ColumnNames2=MyVect_IFIDF.get_feature_names()
        ColumnNames3=MyVect_IFIDF_STEM.get_feature_names()
        ColumnNames4=MyVect_Bernoulli.get_feature_names()
        ColumnNames5=MyVect.get_feature_names()
        #print("Column names: ", ColumnNames2)
        #Create a name
          
    builderS=pd.DataFrame(X1.toarray(),columns=ColumnNames1)
    builderT=pd.DataFrame(X2.toarray(),columns=ColumnNames2)
    builderTS=pd.DataFrame(X3.toarray(),columns=ColumnNames3)
    builderB=pd.DataFrame(X4.toarray(),columns=ColumnNames4)
    builder=pd.DataFrame(X5.toarray(),columns=ColumnNames5)
   
    ## Add column
    #print("Adding new column....")
    
    builderS["Label"]=name
    builderT["Label"]=name
    builderTS["Label"]=name
    builderB["Label"]=name
    builder["Label"]=name
    #print(cnt)
    #cnt+=1
    #print(builderS)
   
    FinalDF_STEM= FinalDF_STEM.append(builderS)
    FinalDF_TFIDF= FinalDF_TFIDF.append(builderT)
    FinalDF_TFIDF_STEM= FinalDF_TFIDF_STEM.append(builderTS)
    FinalDF_Bernoulli=FinalDF_Bernoulli.append(builderB)
    FinalDF=FinalDF.append(builder)
    print(len(FileList))
   
    #print(FinalDF_STEM.head())
FinalDF_STEM=FinalDF_STEM.fillna(0)
FinalDF_TFIDF=FinalDF_TFIDF.fillna(0)
FinalDF_TFIDF_STEM=FinalDF_TFIDF_STEM.fillna(0)
FinalDF_Bernoulli=FinalDF_Bernoulli.fillna(0)
FinalDF=FinalDF.fillna(0)

MyList=[]
for col in FinalDF_TFIDF.columns:
    #print(col)
    LogR=col.isdigit()  ## any numbers
    if(LogR==True):
        #print(col)
        MyList.append(str(col))
       
print(MyList)      
FinalDF_TFIDF.drop(MyList, axis=1, inplace=True)

from sklearn.model_selection import train_test_split
TrainDF1, TestDF1 = train_test_split(FinalDF_STEM, test_size=0.3)
#print(FinalDF_STEM)
#print(TrainDF1)
#print(TestDF1)

TrainDF2, TestDF2 = train_test_split(FinalDF_TFIDF, test_size=0.3)
TrainDF3, TestDF3 = train_test_split(FinalDF_TFIDF_STEM, test_size=0.3)
TrainDF4, TestDF4 = train_test_split(FinalDF_Bernoulli, test_size=0.3)
TrainDF5, TestDF5 = train_test_split(FinalDF, test_size=0.3)

Test1Labels=TestDF1["Label"]
print(Test1Labels)

Test2Labels=TestDF2["Label"]
Test3Labels=TestDF3["Label"]
print(Test2Labels)
Test4Labels=TestDF4["Label"]
Test5Labels=TestDF5["Label"]

## remove labels
TestDF1 = TestDF1.drop(["Label"], axis=1)

TestDF2 = TestDF2.drop(["Label"], axis=1)
TestDF3 = TestDF3.drop(["Label"], axis=1)
print(TestDF1)
TestDF4 = TestDF4.drop(["Label"], axis=1)
TestDF5 = TestDF5.drop(["Label"], axis=1)

## TRAIN ----------------------------
Train1Labels=TrainDF1["Label"]

Train2Labels=TrainDF2["Label"]
Train3Labels=TrainDF3["Label"]
Train4Labels=TrainDF4["Label"]
Train5Labels=TrainDF5["Label"]
print(Train2Labels)
## remove labels
TrainDF1 = TrainDF1.drop(["Label"], axis=1)

TrainDF2 = TrainDF2.drop(["Label"], axis=1)
TrainDF3 = TrainDF3.drop(["Label"], axis=1)
print(TrainDF3)
TrainDF4 = TrainDF4.drop(["Label"], axis=1)
TrainDF5 = TrainDF5.drop(["Label"], axis=1)

from sklearn.naive_bayes import MultinomialNB
MyModelNB1= MultinomialNB()


MyModelNB2= MultinomialNB()
MyModelNB3= MultinomialNB()
MyModelNB4= MultinomialNB()
## When you look up this model, you learn that it wants the

## Run on all three Dfs.................
MyModelNB1.fit(TrainDF1, Train1Labels)

MyModelNB2.fit(TrainDF2, Train2Labels)
MyModelNB3.fit(TrainDF3, Train3Labels)
MyModelNB4.fit(TrainDF5, Train5Labels)

Prediction1 = MyModelNB1.predict(TestDF1)
Prediction2 = MyModelNB2.predict(TestDF2)
Prediction3 = MyModelNB3.predict(TestDF3)
Prediction4 = MyModelNB4.predict(TestDF5)

print("\nThe prediction from NB is:")
print(Prediction1)
print("\nThe actual labels are:")
print(Test1Labels)

print("\nThe prediction from NB is:")
print(Prediction2)
print("\nThe actual labels are:")
print(Test2Labels)

print("\nThe prediction from NB is:")
print(Prediction3)
print("\nThe actual labels are:")
print(Test3Labels)


print("\nThe prediction from NB is:")
print(Prediction4)
print("\nThe actual labels are:")
print(Test5Labels)

## confusion matrix
from sklearn.metrics import confusion_matrix
## The confusion matrix is square and is labels X labels
## We ahve two labels, so ours will be 2X2
#The matrix shows
## rows are the true labels
## columns are predicted
## it is al[habetical
## The numbers are how many
cnf_matrix1 = confusion_matrix(Test1Labels, Prediction1)
print("\nThe confusion matrix is:")
print(cnf_matrix1)

cnf_matrix2 = confusion_matrix(Test2Labels, Prediction2)
print("\nThe confusion matrix is:")
print(cnf_matrix2)

cnf_matrix3 = confusion_matrix(Test3Labels, Prediction3)
print("\nThe confusion matrix is:")
print(cnf_matrix3)

cnf_matrix4 = confusion_matrix(Test5Labels, Prediction4)
print("\nThe confusion matrix is:")
print(cnf_matrix4)
### prediction probabilities
## columns are the labels in alphabetical order
## The decinal in the matrix are the prob of being
## that label
#print(np.round(MyModelNB1.predict_proba(TestDF1),2))
#print(np.round(MyModelNB2.predict_proba(TestDF2),2))
#print(np.round(MyModelNB3.predict_proba(TestDF3),2))

from sklearn.naive_bayes import BernoulliNB
BernModel = BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
BernModel.fit(TrainDF4, Train4Labels)

print("\nBernoulli prediction:\n", BernModel.predict(TestDF4))
print("\nActual:")
print(Test4Labels)
#
bn_matrix = confusion_matrix(Test4Labels, BernModel.predict(TestDF4))
print("\nThe confusion matrix for text Bernoulli is:")
print(bn_matrix)

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
## conda install python-graphviz
## restart kernel (click the little red x next to the Console)
import graphviz
MyDT=DecisionTreeClassifier(criterion='entropy', ##"entropy" or "gini"
                            splitter='best',  ## or "random" or "best"
                            max_depth=None,
                            min_samples_split=2,
                            min_samples_leaf=1,
                            min_weight_fraction_leaf=0.0,
                            max_features=None,
                            random_state=None,
                            max_leaf_nodes=None,
                            min_impurity_decrease=0.0,
                            min_impurity_split=None,
                            class_weight=None)

##
MyDT.fit(TrainDF1, Train1Labels)

#tree.plot_tree(MyDT)
#plt.savefig(temp1)

feature_names=TrainDF1.columns
Tree_Object = tree.export_graphviz(MyDT, out_file=None,
                    ## The following creates TrainDF.columns for each
                    ## which are the feature names.
                      feature_names=feature_names,  
                      class_names=["Alexander", "Hamilton"],
                      #["food","sports","bitcoin"],  
                      filled=True, rounded=True,  
                      special_characters=True)      
                             
graph = graphviz.Source(Tree_Object)
   
graph.render("MyTree1")

from sklearn.model_selection import cross_val_score
scores6 = cross_val_score(MyDT, TrainDF1, Train1Labels, cv=5)

## COnfusion Matrix
print("Prediction\n")
DT_pred=MyDT.predict(TestDF1)
print(len(DT_pred))
print(len(Train1Labels))

dt_matrix1 = confusion_matrix(Test1Labels, DT_pred)
print("\nThe confusion matrix is:")
print(dt_matrix1)


FeatureImp=MyDT.feature_importances_  
indices = np.argsort(FeatureImp)[::-1]
## print out the important features.....
for f in range(TrainDF1.shape[1]):
    if FeatureImp[indices[f]] > 0:
        print("%d. feature %d (%f)" % (f + 1, indices[f], FeatureImp[indices[f]]))
        print ("feature name: ", feature_names[indices[f]])
        
MyDT.fit(TrainDF2, Train2Labels)

#tree.plot_tree(MyDT)
#plt.savefig(temp1)

feature_names=TrainDF2.columns
Tree_Object = tree.export_graphviz(MyDT, out_file=None,
                    ## The following creates TrainDF.columns for each
                    ## which are the feature names.
                      feature_names=feature_names,  
                      class_names=["Alexander", "Hamilton"],
                      #["food","sports","bitcoin"],  
                      filled=True, rounded=True,  
                      special_characters=True)      
                             
graph = graphviz.Source(Tree_Object)
   
graph.render("MyTree2")

from sklearn.model_selection import cross_val_score
scores7 = cross_val_score(MyDT, TrainDF2, Train2Labels, cv=5)

## COnfusion Matrix
print("Prediction\n")
DT_pred=MyDT.predict(TestDF2)
print(len(DT_pred))
print(len(Train2Labels))

dt_matrix2 = confusion_matrix(Test2Labels, DT_pred)
print("\nThe confusion matrix is:")
print(dt_matrix2)


FeatureImp=MyDT.feature_importances_  
indices = np.argsort(FeatureImp)[::-1]
## print out the important features.....
for f in range(TrainDF2.shape[1]):
    if FeatureImp[indices[f]] > 0:
        print("%d. feature %d (%f)" % (f + 1, indices[f], FeatureImp[indices[f]]))
        print ("feature name: ", feature_names[indices[f]])
        
MyDT.fit(TrainDF3, Train3Labels)

#tree.plot_tree(MyDT)
#plt.savefig(temp1)

feature_names=TrainDF3.columns
Tree_Object = tree.export_graphviz(MyDT, out_file=None,
                    ## The following creates TrainDF.columns for each
                    ## which are the feature names.
                      feature_names=feature_names,  
                      class_names=["Alexander", "Hamilton"],
                      #["food","sports","bitcoin"],  
                      filled=True, rounded=True,  
                      special_characters=True)      
                             
graph = graphviz.Source(Tree_Object)
   
graph.render("MyTree3")

from sklearn.model_selection import cross_val_score
scores8 = cross_val_score(MyDT, TrainDF3, Train3Labels, cv=5)

## COnfusion Matrix
print("Prediction\n")
DT_pred=MyDT.predict(TestDF3)
print(len(DT_pred))
print(len(Train3Labels))

dt_matrix3 = confusion_matrix(Test3Labels, DT_pred)
print("\nThe confusion matrix is:")
print(dt_matrix3)


FeatureImp=MyDT.feature_importances_  
indices = np.argsort(FeatureImp)[::-1]
## print out the important features.....
for f in range(TrainDF3.shape[1]):
    if FeatureImp[indices[f]] > 0:
        print("%d. feature %d (%f)" % (f + 1, indices[f], FeatureImp[indices[f]]))
        print ("feature name: ", feature_names[indices[f]])

MyDT.fit(TrainDF4, Train4Labels)

#tree.plot_tree(MyDT)
#plt.savefig(temp1)

feature_names=TrainDF4.columns
Tree_Object = tree.export_graphviz(MyDT, out_file=None,
                    ## The following creates TrainDF.columns for each
                    ## which are the feature names.
                      feature_names=feature_names,  
                      class_names=["Alexander", "Hamilton"],
                      #["food","sports","bitcoin"],  
                      filled=True, rounded=True,  
                      special_characters=True)      
                             
graph = graphviz.Source(Tree_Object)
   
graph.render("MyTree4")

from sklearn.model_selection import cross_val_score
scores9 = cross_val_score(MyDT, TrainDF4, Train4Labels, cv=5)

## COnfusion Matrix
print("Prediction\n")
DT_pred=MyDT.predict(TestDF4)
print(len(DT_pred))
print(len(Train4Labels))

dt_matrix4 = confusion_matrix(Test4Labels, DT_pred)
print("\nThe confusion matrix is:")
print(dt_matrix4)


FeatureImp=MyDT.feature_importances_  
indices = np.argsort(FeatureImp)[::-1]
## print out the important features.....
for f in range(TrainDF4.shape[1]):
    if FeatureImp[indices[f]] > 0:
        print("%d. feature %d (%f)" % (f + 1, indices[f], FeatureImp[indices[f]]))
        print ("feature name: ", feature_names[indices[f]])
        
MyDT.fit(TrainDF5, Train5Labels)

#tree.plot_tree(MyDT)
#plt.savefig(temp1)

feature_names=TrainDF5.columns
Tree_Object = tree.export_graphviz(MyDT, out_file=None,
                    ## The following creates TrainDF.columns for each
                    ## which are the feature names.
                      feature_names=feature_names,  
                      class_names=["Alexander", "Hamilton"],
                      #["food","sports","bitcoin"],  
                      filled=True, rounded=True,  
                      special_characters=True)      
                             
graph = graphviz.Source(Tree_Object)
   
graph.render("MyTree5")

from sklearn.model_selection import cross_val_score
scores10 = cross_val_score(MyDT, TrainDF5, Train5Labels, cv=5)

## COnfusion Matrix
print("Prediction\n")
DT_pred=MyDT.predict(TestDF5)
print(len(DT_pred))
print(len(Train5Labels))

dt_matrix5 = confusion_matrix(Test5Labels, DT_pred)
print("\nThe confusion matrix is:")
print(dt_matrix5)


FeatureImp=MyDT.feature_importances_  
indices = np.argsort(FeatureImp)[::-1]
## print out the important features.....
for f in range(TrainDF5.shape[1]):
    if FeatureImp[indices[f]] > 0:
        print("%d. feature %d (%f)" % (f + 1, indices[f], FeatureImp[indices[f]]))
        print ("feature name: ", feature_names[indices[f]])
        
from sklearn.model_selection import cross_val_score

scores1 = cross_val_score(MyModelNB1, TrainDF1, Train1Labels, cv=5)
scores2 = cross_val_score(MyModelNB2, TrainDF2, Train2Labels, cv=5)
scores3 = cross_val_score(MyModelNB3, TrainDF3, Train3Labels, cv=5)
scores4 = cross_val_score(BernModel, TrainDF4, Train4Labels, cv=5)
scores5 = cross_val_score(MyModelNB4, TrainDF5, Train5Labels, cv=5)

print()
print("Score 1: %0.2f accuracy with a standard deviation of %0.2f" % (scores1.mean(), scores1.std()))
print("Score 2: %0.2f accuracy with a standard deviation of %0.2f" % (scores2.mean(), scores2.std()))
print("Score 3: %0.2f accuracy with a standard deviation of %0.2f" % (scores3.mean(), scores3.std()))
print("Score 4: %0.2f accuracy with a standard deviation of %0.2f" % (scores4.mean(), scores4.std()))
print("Score 5: %0.2f accuracy with a standard deviation of %0.2f" % (scores5.mean(), scores5.std()))
print("Score 6: %0.2f accuracy with a standard deviation of %0.2f" % (scores6.mean(), scores6.std()))
print("Score 7: %0.2f accuracy with a standard deviation of %0.2f" % (scores7.mean(), scores7.std()))
print("Score 8: %0.2f accuracy with a standard deviation of %0.2f" % (scores8.mean(), scores8.std()))
print("Score 9: %0.2f accuracy with a standard deviation of %0.2f" % (scores9.mean(), scores9.std()))
print("Score 10: %0.2f accuracy with a standard deviation of %0.2f" % (scores10.mean(), scores10.std()))

from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

SVM_Model=LinearSVC(C=10)

SVM_Model.fit(TrainDF5, Train5Labels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
print("SVM prediction:\n", SVM_Model.predict(TestDF5))
print("Actual:")
print(Test5Labels)

SVM_matrix1 = confusion_matrix(Test5Labels, SVM_Model.predict(TestDF5))
print("\nThe confusion matrix for basic linear SVC is:")
print(SVM_matrix1)
print("\n\n")

TRAIN= TrainDF5
TRAIN_Labels= Train5Labels
TEST= TestDF5
TEST_Labels= Test5Labels


SVM_Model1=LinearSVC(C=50)
SVM_Model1.fit(TRAIN, TRAIN_Labels)

print("SVM prediction:\n", SVM_Model1.predict(TEST))
print("Actual:")
print(TEST_Labels)

SVM_matrix2 = confusion_matrix(TEST_Labels, SVM_Model1.predict(TEST))
print("\nThe confusion matrix for Linear SVC C=50 is:")
print(SVM_matrix2)
print("\n\n")
#--------------other kernels
## RBF
SVM_Model2=sklearn.svm.SVC(C=10, kernel='rbf',
                           verbose=True, gamma="auto")
SVM_Model2.fit(TRAIN, TRAIN_Labels)

print("SVM prediction:\n", SVM_Model2.predict(TEST))
print("Actual:")
print(TEST_Labels)

SVM_matrix3 = confusion_matrix(TEST_Labels, SVM_Model2.predict(TEST))
print("\nThe confusion matrix for rbf SVM is:")
print(SVM_matrix3)
print("\n\n")

SVM_Model3=sklearn.svm.SVC(C=10, kernel='poly',degree=3,
                           gamma="auto", verbose=True)

print(SVM_Model3)
SVM_Model3.fit(TRAIN, TRAIN_Labels)
print("SVM prediction:\n", SVM_Model3.predict(TEST))
print("Actual:")
print(TEST_Labels)

SVM_matrix4 = confusion_matrix(TEST_Labels, SVM_Model3.predict(TEST))
print("\nThe confusion matrix for SVM poly d=2  is:")
print(SVM_matrix4)
print("\n\n")

import matplotlib.pyplot as plt
## Credit: https://medium.com/@aneesha/visualising-top-features-in-linear-svm-with-scikit-learn-and-matplotlib-3454ab18a14d
## Define a function to visualize the TOP words (variables)
def plot_coefficients(MODEL=SVM_Model, COLNAMES=TrainDF5.columns, top_features=10):
    ## Model if SVM MUST be SVC, RE: SVM_Model=LinearSVC(C=10)
    coef = MODEL.coef_.ravel()
    top_positive_coefficients = np.argsort(coef,axis=0)[-top_features:]
    top_negative_coefficients = np.argsort(coef,axis=0)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    colors = ["red" if c < 0 else "blue" for c in coef[top_coefficients]]
    plt.bar(  x=  np.arange(2 * top_features)  , height=coef[top_coefficients], width=.5,  color=colors)
    feature_names = np.array(COLNAMES)
    plt.xticks(np.arange(0, (2*top_features)), feature_names[top_coefficients], rotation=60, ha="right")
    plt.show()
   

plot_coefficients()
#plt.savefig('KeyWords.pdf')

from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import ward, dendrogram

#My_KMean= KMeans(n_clusters=3)
#My_KMean.fit(My_Orig_DF)
#My_labels=My_KMean.predict(My_Orig_DF)
#print(My_labels)
kmeanDF= FinalDF
kmeanDF_labels = FinalDF['Label']
#FinalDF_TFIDF.drop(MyList, axis=1, inplace=True)
kmeanDF.drop('Label', axis = 1, inplace = True)

from sklearn import preprocessing
#from sklearn.cluster import KMeans
import seaborn as sns

My_KMean2 = KMeans(n_clusters=2).fit(preprocessing.normalize(kmeanDF))
My_KMean2.fit(kmeanDF)
My_labels2=My_KMean2.predict(kmeanDF)
#print(My_labels2)
#print(Labels_DF)

Labels_DF_KMeans = kmeanDF_labels
Labels_DF_KMeans['Predictions'] = My_labels2
#Labels_DF_KMeans.loc[(Labels_DF_KMeans.Predictions == 0),'Predictions']='Alexander'
#Labels_DF_KMeans.loc[(Labels_DF_KMeans.Predictions == 1),'Predictions']='Hamilton'



print(Labels_DF_KMeans)

#My_KMean3= KMeans(n_clusters=3)
#My_KMean3.fit(My_Orig_DF)
#My_labels3=My_KMean3.predict(My_Orig_DF)
#print("Silhouette Score for k = 3 \n",silhouette_score(My_Orig_DF, My_labels3))


#https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
#length of the document: called cosine similarity
cosdist = 1 - cosine_similarity(kmeanDF)
#print(cosdist)
#print(np.round(cosdist,3))  #cos dist should be .02

#----------------------------------------------------------
## Hierarchical Clustering using ward and cosine sim
linkage_matrix = ward(cosdist) #define the linkage_matrix
#using ward clustering pre-computed distances
#print(linkage_matrix)
fig = plt.figure(figsize=(25, 10))
#dn = dendrogram(linkage_matrix)
#print(len(Labels_DF_KMeans))
#dn = dendrogram(
#    linkage_matrix,
#    labels = list(Labels_DF_KMeans),
#    leaf_rotation = 90,
#    leaf_font_size = 6
#)
dendoLabels = list(Labels_DF_KMeans)
dendoLabels.pop()
dendoLabels2 = []
cntA = 1
cntB = 1
for i in dendoLabels:
    numA = cntA
    numA = str(numA)
    numB = cntB
    numB = str(numB)
    i = i.split()
    i = i[0]
    if i == 'Alexander':
        i = i + '' + numA
        dendoLabels2.append(i)
        cntA+=1
    else:
        i = i + '' + numB
        dendoLabels2.append(i)
        cntB+=1
dn = dendrogram(linkage_matrix, leaf_rotation=90, leaf_font_size=8, labels=dendoLabels2)
#labels=kmeanDF.index
plt.show()

scores11 = cross_val_score(SVM_Model, TrainDF5, Train5Labels, cv=5)
scores12 = cross_val_score(SVM_Model1, TRAIN, TRAIN_Labels, cv=5)
scores13 = cross_val_score(SVM_Model2, TRAIN, TRAIN_Labels, cv=5)
scores14 = cross_val_score(SVM_Model3, TRAIN, TRAIN_Labels, cv=5)

print()
print("Score 11: %0.2f accuracy with a standard deviation of %0.2f" % (scores11.mean(), scores11.std()))
print("Score 12: %0.2f accuracy with a standard deviation of %0.2f" % (scores12.mean(), scores12.std()))
print("Score 13: %0.2f accuracy with a standard deviation of %0.2f" % (scores13.mean(), scores13.std()))
print("Score 14: %0.2f accuracy with a standard deviation of %0.2f" % (scores14.mean(), scores14.std()))

alexanderList = ['Alexander Clean/Alexander 22.txt', 'Alexander Clean/Alexander 36.txt', 'Alexander Clean/Alexander 37.txt', 'Alexander Clean/Alexander 23.txt', 'Alexander Clean/Alexander 35.txt', 'Alexander Clean/Alexander 21.txt', 'Alexander Clean/Alexander 20.txt', 'Alexander Clean/Alexander 34.txt', 'Alexander Clean/Alexander 18.txt', 'Alexander Clean/Alexander 30.txt', 'Alexander Clean/Alexander 24.txt', 'Alexander Clean/Alexander 9.txt', 'Alexander Clean/Alexander 8.txt', 'Alexander Clean/Alexander 25.txt', 'Alexander Clean/Alexander 31.txt', 'Alexander Clean/Alexander 19.txt', 'Alexander Clean/Alexander 27.txt', 'Alexander Clean/Alexander 33.txt', 'Alexander Clean/Alexander 32.txt', 'Alexander Clean/Alexander 26.txt', 'Alexander Clean/Alexander 41.txt', 'Alexander Clean/Alexander 40.txt', 'Alexander Clean/Alexander 42.txt', 'Alexander Clean/Alexander 43.txt', 'Alexander Clean/Alexander 46.txt', 'Alexander Clean/Alexander 44.txt', 'Alexander Clean/Alexander 45.txt', 'Alexander Clean/Alexander 17.txt', 'Alexander Clean/Alexander 6.txt', 'Alexander Clean/Alexander 7.txt', 'Alexander Clean/Alexander 16.txt', 'Alexander Clean/Alexander 14.txt', 'Alexander Clean/Alexander 28.txt', 'Alexander Clean/Alexander 5.txt', 'Alexander Clean/Alexander 4.txt', 'Alexander Clean/Alexander 29.txt', 'Alexander Clean/Alexander 15.txt', 'Alexander Clean/Alexander 39.txt', 'Alexander Clean/Alexander 11.txt', 'Alexander Clean/Alexander 1.txt', 'Alexander Clean/Alexander 10.txt', 'Alexander Clean/Alexander 38.txt', 'Alexander Clean/Alexander 12.txt', 'Alexander Clean/Alexander 3.txt', 'Alexander Clean/Alexander 2.txt', 'Alexander Clean/Alexander 13.txt']
FileList += alexanderList
CorpusDF_DH=pd.DataFrame()
Vect_DH = MyVect.fit_transform(FileList)
ColumnNamesLDA_DH=MyVect.get_feature_names()
CorpusDF_DH=pd.DataFrame(Vect_DH.toarray(),columns=ColumnNamesLDA_DH)
#print(CorpusDF_DH)

from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
lda_model_DH = LatentDirichletAllocation(n_components=10, max_iter=10000, learning_method='online')
#lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online')
LDA_DH_Model = lda_model_DH.fit_transform(Vect_DH)

print("SIZE: ", LDA_DH_Model.shape)  # (NO_DOCUMENTS, NO_TOPICS)

# Let's see how the first document in the corpus looks like in
## different topic spaces
print("First Doc in Alexander and Hamilton data...")
print(LDA_DH_Model[0])
print("Seventh Doc in Alexander and Hamilton...")
print(LDA_DH_Model[19])

def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])
        
## Print LDA using print function from above
print("LDA Alexander and Hamilton Model:")
print_topics(lda_model_DH, MyVect)

import pyLDAvis.sklearn as LDAvis
import pyLDAvis
pyLDAvis.enable_notebook()
## conda install -c conda-forge pyldavis
#pyLDAvis.enable_notebook() ## not using notebook
panel = LDAvis.prepare(lda_model_DH,
                       Vect_DH, MyVect, mds='tsne')
pyLDAvis.show(panel, local=False)

from sklearn.decomposition import LatentDirichletAllocation
NUM_TOPICS=10
lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10000, learning_method='online')
#lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online')
   
lda_Z_DF = lda_model.fit_transform(CorpusDF_DH)
print(lda_Z_DF.shape)  # (NO_DOCUMENTS, NO_TOPICS)

def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                    for i in topic.argsort()[:-top_n - 1:-1]])
 
print("LDA Model:")
print_topics(lda_model, MyVect)

word_topic = np.array(lda_model.components_)
#print(word_topic)
word_topic = word_topic.transpose()

num_top_words = 20
ColumnNames=MyVect.get_feature_names()
#print(type(ColumnNames))
vocab_array = np.asarray(ColumnNames)

#fontsize_base = 70 / np.max(word_topic) # font size for word with largest share in corpus
fontsize_base = 10

for t in range(NUM_TOPICS):
    plt.subplot(1, NUM_TOPICS, t + 1)  # plot numbering starts with 1
    plt.subplots_adjust(left = 3, right = 5)  # plot numbering starts with 1
    plt.ylim(0, num_top_words + 0.5)  # stretch the y-axis to accommodate the words
    plt.xlim(right=10)  # stretch the y-axis to accommodate the words
    plt.xticks([])  # remove x-axis markings ('ticks')
    plt.yticks([]) # remove y-axis markings ('ticks')
    plt.title('Topic #{}'.format(t))
    top_words_idx = np.argsort(word_topic[:,t])[::-1]  # descending order
    top_words_idx = top_words_idx[:num_top_words]
    top_words = vocab_array[top_words_idx]
    top_words_shares = word_topic[top_words_idx, t]
    for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
        plt.text(0.3, num_top_words-i-0.5, word, fontsize=fontsize_base)
        
import seaborn as sns
sns.heatmap(cnf_matrix1, annot=True)
plt.show()
sns.heatmap(cnf_matrix2, annot=True)
plt.show()
sns.heatmap(cnf_matrix3, annot=True)
plt.show()
sns.heatmap(cnf_matrix4, annot=True)
plt.show()
sns.heatmap(bn_matrix, annot=True)
plt.show()
sns.heatmap(dt_matrix1, annot=True)
plt.show()
sns.heatmap(dt_matrix2, annot=True)
plt.show()
sns.heatmap(dt_matrix3, annot=True)
plt.show()
sns.heatmap(dt_matrix4, annot=True)
plt.show()
sns.heatmap(dt_matrix5, annot=True)
plt.show()
sns.heatmap(SVM_matrix1, annot=True)
plt.show()
sns.heatmap(SVM_matrix2, annot=True)
plt.show()
sns.heatmap(SVM_matrix3, annot=True)
plt.show()
sns.heatmap(SVM_matrix4, annot=True)
plt.show()



    
    
    
    
    
