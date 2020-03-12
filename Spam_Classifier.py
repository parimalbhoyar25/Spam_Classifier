#  Importing the Dataset

import pandas as pd

messages=pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t',
                     names=["label","message"])

#Data Cleaning and Preprocessing

import re     #regural expression library
import nltk

from nltk.corpus import stopwords    #stopwords for removing unwanted words
from nltk.stem import PorterStemmer  #for performing stemming
ps=PorterStemmer()                   #creating object
corpus=[]                            #creating empty list

for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]',' ',messages['message'][i])  #for removing all the characters except A to Z from each sentences
    review = review.lower()                                  #to lower case every words
    review = review.split()                                  #to split the sentenses in list of words
    
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]#if word not present in stopword 
    #of english then applying stemming on that word to get base word of the word
    review = ' '.join(review)   #joining all list of words to make a sentence
    corpus.append(review)       #adding it to new list corpus

#Creating the Bag of Words Model
    
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000)
X=cv.fit_transform(corpus).toarray() 

#creating dummy varibale
y=pd.get_dummies(messages['label'])   
y=y.iloc[:,1].values   #selecting one column

#Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.20, random_state=0)

#Training model using Naive Bayes classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model= MultinomialNB().fit(X_train,y_train)

y_pred=spam_detect_model.predict(X_test)

#creating confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred)     #we get a good accuracy score of 0.9847533
