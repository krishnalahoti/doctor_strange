# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 02:31:14 2020

@author: User
"""

import numpy as np
import pandas as pd

df = pd.read_csv('../TextFiles/moviereviews.tsv',sep = '\t')

df.head()

df.isnull().sum()

df.dropna(inplace=True)

df.isnull().sum()
### many times people just leave a review as blank ,in tech terms as whitespace.

# we need to remove such reviews too.
blanks = []

for i,lab,rv in df.itertuples():
    if rv.isspace():
        blanks.append(i)

print(blanks)

df.drop(blanks,inplace=True)

df['label'].value_counts()

len(df)

# lets perofrm a train_test_split.
X = df['review']
y = df['label']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

from sklearn.svm import LinearSVC

clf = LinearSVC()

# now make use of pipeline to perform the above operations for train and test set.


from sklearn.pipeline import Pipeline

text_clf = Pipeline([('tfidf',vectorizer),('clf',clf)])

text_clf.fit(X_train,y_train)


y_pred = text_clf.predict(X_test)


from sklearn.metrics import confusion_matrix,classification_report


con_df = pd.DataFrame(confusion_matrix(y_test,y_pred),index=['neg','pos'],columns=['neg','pos'])

classification_report(y_test,y_pred)
















