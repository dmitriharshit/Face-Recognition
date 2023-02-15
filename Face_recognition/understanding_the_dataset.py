import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

olivetti_data = fetch_olivetti_faces()

# there are 400 images(each person has 10 images and there are 10 people) and each image is 64x64
features = olivetti_data.data
# we represent target variables (people) with integers (face ids)
target = olivetti_data.target

fig, subplot = plt.subplots(nrows=5, ncols=8, figsize=(14, 8))
subplot = subplot.flatten()
# print(subplot)

# to see unique photos of all 40 people :
for uniqueuserid in np.unique(target):
    image_index = uniqueuserid*8
    subplot[uniqueuserid].imshow(
        features[image_index].reshape(64, 64), cmap='gray')
    subplot[uniqueuserid].set_xticks([])
    subplot[uniqueuserid].set_yticks([])
    subplot[uniqueuserid].set_title('Face id=%s' % uniqueuserid)

plt.suptitle('The dataset(40 people)')
plt.show()


# to see all 10 photo of a single person
fig, subplot = plt.subplots(nrows=1, ncols=10, figsize=(14, 8))
for uniqueuserid in range(10):
    image_index = uniqueuserid
    subplot[uniqueuserid].imshow(
        features[image_index].reshape(64, 64), cmap='gray')
    subplot[uniqueuserid].set_xticks([])
    subplot[uniqueuserid].set_yticks([])
    subplot[uniqueuserid].set_title('photo no%s' % uniqueuserid)

plt.suptitle('10 photos of single person')
plt.show()
