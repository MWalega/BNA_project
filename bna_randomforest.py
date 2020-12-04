# -*- coding: utf-8 -*-
"""BNA_RandomForest.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1m5HyzIgOQx6c8LY0Sqpr6qJRvlPVRzOi
"""

import torch
from torch.utils.data import random_split
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('/content/BankNote_Authentication.csv')

df.head()

# independent and dependent features
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

classifier = RandomForestClassifier()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

score = accuracy_score(y_test,y_pred)
score

pickle_out = open('classifier.pkl', 'wb')
pickle.dump(classifier, pickle_out)
pickle_out.close()

classifier.predict([[2,3,4,1]])