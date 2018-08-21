import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from .main import HOG
from ..feature_extraction.main import extract_features


hog = HOG()

images = glob('../../../../test_images/*.jpg')
cars = []
notcars = []

car_features = extract_features(cars)