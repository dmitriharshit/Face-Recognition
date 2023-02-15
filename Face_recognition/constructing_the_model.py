import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn import metrics

olivetti_data = fetch_olivetti_faces()

# there are 400 images(each person has 10 images and there are 10 people) and each image is 64x64
features = olivetti_data.data
# we represent target variables (people) with integers (face ids)
target = olivetti_data.target

feature_train, feature_test, target_train,  target_test = train_test_split(
    features, target, test_size=0.25, stratify=target, random_state=0)

# from previous we know optimal number of components=100
pca = PCA(n_components=100, whiten=True)
pca.fit(feature_train)

feature_train_pca = pca.transform(feature_train)
feature_test_pca = pca.transform(feature_test)

# models:
models = [('Logistic Regression', LogisticRegression()),
          ('Support Vector Machine', SVC()), ('Naive Bayes Classifier', GaussianNB())]

for name, model in models:
    classifier_model = model
    classifier_model.fit(feature_train_pca, target_train)
    predictions = classifier_model.predict(feature_test_pca)

    print(' %s : accuracy score %s' % (name,
          (metrics.accuracy_score(target_test, predictions))))
