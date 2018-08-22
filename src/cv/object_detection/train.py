import glob
import cv2
import numpy as np
import pickle
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from feature_extraction.main import extract_hog_features

def load_train_data(colorspace, orient, pix_per_cell, cell_per_block, hog_channel):
    car_images_src = '../../../train_data/vehicles/**/*.png'
    not_car_images_src = '../../../train_data/non-vehicles/**/*.png'
    car_images = glob.glob(car_images_src)
    not_car_images = glob.glob(not_car_images_src)

    car_features = extract_hog_features(car_images, colorspace, orient, pix_per_cell, cell_per_block, hog_channel)
    not_car_features = extract_hog_features(not_car_images, colorspace, orient, pix_per_cell, cell_per_block, hog_channel)

    X = np.vstack((car_features, not_car_features)).astype(np.float64)

    y = np.hstack((np.ones(len(car_features)), np.zeros(len(not_car_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)
    X_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to X
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)
    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    return X_train, y_train, X_test, y_test, X_scaler

def train(colorspace, orientations, pixel_per_cell, cell_per_block, hog_channel, load):
    modelPath = 'svc_classifier.pkl'
    if (not load):
        svc = LinearSVC()

        X_train, y_train, X_test, y_test, X_scaler = load_train_data(colorspace, orientations, pixel_per_cell, cell_per_block,
                                                           hog_channel)

        svc.fit(X_train, y_train)

        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
        with open(modelPath, 'wb') as fid:
            pickle.dump(svc, fid)
            print('Save model to', modelPath)
    else:
        with open(modelPath, 'rb') as fid:
            svc = pickle.load(fid)

    return svc, X_scaler