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

# from previous we know optimal number of components=100
pca = PCA(n_components=100, whiten=True)
pca.fit(feature_train)

feature_train_pca = pca.transform(feature_train)
target_train_pca = pca.transform(target_train)


# after we find the optimal 100 PC we can check 'eigenfaces'
no_of_eigenfaces = len(pca.components_)
eigen_faces = pca.components_.reshape((no_of_eigenfaces, 64, 64))

fig, subplot = plt.subplots(nrows=10, ncols=10, figsize=(15, 15))
subplot = subplot.flatten()

for i in range(no_of_eigenfaces):
    subplot[i].imshow(eigen_faces[i], cmap='gray')
    subplot[i].set_xticks([])
    subplot[i].set_yticks([])

plt.show()
