import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
import numpy as np
import cv2
import sys
sys.path.append('../')
from bin_spatial.main import bin_spatial
from color_hist.main import  color_hist
from hog.main import HOG

def data_look(car_list, notcar_list):
    data_dict = {}
    data_dict["n_cars"] = len(car_list)
    data_dict["n_nocars"] = len(notcar_list)

    example_img = mpimg.imread(car_list[0])

    data_dict["image_shape"] = example_img.shape
    data_dict["data_type"] = example_img.dtype

    return data_dict

def combine_features(features_list):
    return np.vstack(features_list).astype(np.float64)

def normalize_data(data):
    data_scaler = StandardScaler().fit(data)
    scaled_data = data_scaler.transform(data)
    return scaled_data

def extract_spatial_color_hist_features(imgs, cspace='RGB', spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256)):
    features = []
    for filePath in imgs:
        img = cv2.imread(filePath)
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2YUV)
        else:
            feature_image = np.copy(img_RGB)

        spatial_features = bin_spatial(feature_image, spatial_size)
        hist_features = color_hist(feature_image, hist_bins, hist_range)
        features.append(np.concatenate(spatial_features, hist_features))
        return features

def extract_hog_features(imgs, cspace='RGB', orient=9, pixel_per_cell=8, cell_per_block=2, hog_channel=0):
    features = []
    for imgSrc in imgs:
        img = cv2.imread(imgSrc)
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2YUV)
        else:
            feature_image = np.copy(img_RGB)

        hog = HOG(orient, pixel_per_cell, cell_per_block)
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                channel_hog_feature = hog.get_hog_features(feature_image[:,:,channel], False, True)
                hog_features.append(channel_hog_feature)
            hog_feature = np.ravel(hog_features)
            features.append(hog_features)
        else:
            feature = hog.get_hog_features(feature_image, False, True)
            features.append(feature)

    return features
