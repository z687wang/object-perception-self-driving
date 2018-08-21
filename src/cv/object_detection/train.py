import glob
import cv2
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

from feature_extraction.main import extract_hog_features

def load_train_data():
    colorspace = 'YUV'
    orient = 11
    pix_per_cell = 16
    cell_per_block = 2
    hog_channel = 'ALL'

    car_images_src = '../../../train_data/vehicles/**/*.png'
    not_car_images_src = '../../../train_data/non-vehicles/**/*.png'
    car_images = glob.glob(car_images_src)
    not_car_images = glob.glob(not_car_images_src)

    car_features = extract_hog_features(car_images, colorspace, orient, pix_per_cell, cell_per_block, hog_channel)
    not_car_features = extract_hog_features(not_car_images, colorspace, orient, pix_per_cell, cell_per_block, hog_channel)

    X = np.vstack((car_features, not_car_features)).astype(np.float64)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(not_car_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)
    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    return X_train, y_train, X_test, y_test

def train():
    svc = LinearSVC()

    X_train, y_train, X_test, y_test = load_train_data()

    svc.fit(X_train, y_train)

    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    return svc