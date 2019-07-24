import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os
import pickle
import pandas

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
                for img in f:
                    img = cv2.imread(r+'/'+img)
                    img = cv2.resize(img, (100, 100))
                    images.append(img)
                    labels.append(CLASSIFY[r.split('/')[2]])

    return np.array(images), labels


def hogDescriptor():
    winSize = (20, 20)
    blockSize = (10, 10)
    blockStride = (5, 5)
    cellSize = (10, 10)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    useSignedGradients = True

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize,
                            nbins, derivAperture, winSigma, histogramNormType,
                            L2HysThreshold, gammaCorrection, nlevels,
                            useSignedGradients)
    return hog


def getFeature():
    """
        **get data from pkl**

        :param none: none

        :return images, labels
        :rtype numpy, list
    """

    descriptions = []

    hog = hogDescriptor()

    with open('images.pkl', 'rb') as file:
        unpickler = pickle.Unpickler(file)
        images = unpickler.load()

    with open('labels.pkl', 'rb') as file:
        unpickler = pickle.Unpickler(file)
        labels = unpickler.load()

    for image in images:
        descript = hog.compute(image)
        xi = [float(x[0]) for x in descript]
        descriptions.append(xi)

    return descriptions, labels


def SVM_Classifycaition(X, y):
    svm = cv2.ml.SVM_create()
    # Set SVM type
    svm.setType(cv2.ml.SVM_C_SVC)
    # Set SVM Kernel to Radial Basis Function (RBF)
    svm.setKernel(cv2.ml.SVM_RBF)
    # Set parameter C
    svm.setC(1)
    # Set parameter Gamma
    svm.setGamma(1)
    # Train SVM on training data
    svm.train(np.float32(X), cv2.ml.ROW_SAMPLE, np.int32(y))
    # Save trained model
    # svm.save("hand.yml")
    print(type(np.float32(X)))
    print(np.float32(X).shape)


def predict(image):

    hog = hogDescriptor()

    svm = cv2.ml.SVM_load('hand.yml')
    descript = hog.compute(image)
    xi = [float(x[0]) for x in descript]

    xi = np.array(xi)
    xi = [xi]

    testResponse = svm.predict(np.float32(xi))[1]
    print(testResponse)

    print(type(np.float32(xi)))
    print(np.float32(xi).shape)


# image = cv2.imread('mau_thu.png')

# image = cv2.resize(image, (100, 100))

# predict(image)
