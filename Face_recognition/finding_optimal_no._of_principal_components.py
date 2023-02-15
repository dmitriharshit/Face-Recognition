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

feature_train, target_train, feature_test, target_test = train_test_split(
    features, target, test_size=0.25, stratify=target, random_state=0)

# lets try to find the optimal number of eigen vectors(principal components)
pca = PCA()
pca.fit(features)

# plotting variance vs components graph
# we want to choose the number of components such that the variance is max

plt.figure(1, figsize=(12, 8))
plt.plot(pca.explained_variance_, linewidth=2)
plt.xlabel('components')
plt.ylabel('explained Variance')
plt.show()
# we can clearly see that after 100 components there is not much change in the variance, so optimal number of components=100
