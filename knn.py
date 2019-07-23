import cv2
import numpy as np
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
                print(r)
                # print(f)
                for img in f:
                    img = cv2.imread(r+'/'+img)
                    img = cv2.resize(img, (100, 100))
                    images.append(img)
                    labels.append(CLASSIFY[r.split('/')[2]])

    return np.array(images), labels


def getFeature():
    """
        **get data from pkl**

        :param none: none

        :return images, labels
        :rtype numpy, list
    """
    with open('images.pkl', 'rb') as file:
        unpickler = pickle.Unpickler(file)
        images = unpickler.load()

    with open('labels.pkl', 'rb') as file:
        unpickler = pickle.Unpickler(file)
        labels = unpickler.load()

    return images, labels


# images, labels = read()

# with open('images.pkl', 'wb') as file:
#     pickle.dump(images, file)

# with open('labels.pkl', 'wb') as file:
#     pickle.dump(labels, file)

# getFeature()
