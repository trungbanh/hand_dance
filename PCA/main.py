import cv2 
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import os
import pickle

CLASSIFY = {
    'fruit': 1,
    'lead': 2,
    'pray': 3,
    'rice': 4
}


def read(path='./knowledge/') -> np:
    """
        **read all image**
        :param path: is a path (defaut is '.knowledge')
        :type: file: str

        :return list of image and class
        :rtype numpy, list
    """

    images = []
    labels = []

    for _, d, _ in os.walk('./knowledge/'):
        for path in d:
            for r, _, f in os.walk('./knowledge/'+path):
                # print(r)
                for img in f:
                    img = cv2.imread(r+'/'+img,0)
                    img = cv2.resize(img, (128, 128))
                    img = np.array(img).flatten()
                    images.append(img)
                    labels.append(CLASSIFY[r.split('/')[2]])

    return np.array(images), labels


data, labels = read()

faces = np.reshape(data,(127, -1))
faces_pca = PCA(n_components=0.8)
faces_pca.fit(faces)

components = faces_pca.transform(faces)
# print(components.shape) # => (127, 23)
projected = faces_pca.inverse_transform(components)

nbrs = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree').fit(components, labels)

test = cv2.imread("lead.png",0)
test = cv2.resize(test, (128,128))
test = np.array(test).flatten()
test = np.reshape(test,(1, -1))

testP = faces_pca.transform(test)
print(test.shape)

print (testP.shape)

res = nbrs.predict(testP)
print (res)

with open('pca.pkl','wb') as file:
    pickle.dump(faces_pca,file)
