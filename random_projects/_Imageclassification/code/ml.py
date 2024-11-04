#   ML-Model
#   From data_functions-py Import vorbereitete Daten.
from data_functions import X, y
from pictures import artist

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


#       Pipeline combining standard-scaler with svm
pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel = "rbf", C = 10))])

pipe.fit(X_train, y_train)
#print(f"Score of SVM after Standard Scaler: {pipe.score(X_test, y_test)}")

#       Grid-method and different learning-models

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

#   dictionary for testing the different learning methods and their Hyperparameters
model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto',probability=True),
        'params' : {
            'svc__C': [1,10,100,1000],
            'svc__kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'randomforestclassifier__n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'logisticregression__C': [1,5,10]
        }
    }
}
#   ausgeklemmt, weil lange dauert und für Applikation nicht relevant. benutzt wird svm.
"""scores = []
best_estimators = {}
for algo, mp in model_params.items():
    pipe = make_pipeline(StandardScaler(), mp['model'])
    clf =  GridSearchCV(pipe, mp['params'], cv=5, return_train_score=False)
    clf.fit(X_train, y_train)
    scores.append({
        'model': algo,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    best_estimators[algo] = clf.best_estimator_

import pandas as pd
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
print(df)"""

#       input of the image to test in the model
import numpy as np
import cv2
from data_functions import w2d  
import os



def guess_image(img):
    img = cv2.imread(img)
    scalled_raw_img = cv2.resize(img, (32,32))
    img_har = w2d(img, "db1", 5)
    scalled_img_har = cv2.resize(img_har, (32,32))
    combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1),scalled_img_har.reshape(32*32,1)))
    combined_img = np.array(combined_img).reshape(1,4096).astype(float)
    return combined_img


#img = "./code/test/14754ccf-4d17-4853-b28f-51cbb5993f13.jpeg"


lr = LogisticRegression()
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)

"""print(np.array(predictions).reshape(1,-1).shape)
print(np.array(y_test).reshape(1,-1).shape)
print(lr.score(np.array(y_test).reshape(-1,1), np.array(predictions).reshape(-1,1)))"""



#prediction1 = pipe.predict(guess_image(img))
#print(f"Using SVM the prediction is {artist[prediction1[0]]}")
#prediction2 = lr.predict(guess_image(img))
#print(f"Using Logistic Regression the prediction is {artist[prediction2[0]]}")