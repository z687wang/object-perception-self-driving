import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.image as mpimg
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
        img = mpimg.imread(filePath)
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(img)

        spatial_features = bin_spatial(feature_image, spatial_size)
        hist_features = color_hist(feature_image, hist_bins, hist_range)
        features.append(np.concatenate(spatial_features, hist_features))
        return features

def extract_hog_features(imgs, cspace='RGB', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0):
    features = []
    hog = HOG(orient, pix_per_cell, cell_per_block)
    for imgSrc in imgs:
        img = mpimg.imread(imgSrc)
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(img)

        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                channel_hog_feature = hog.get_hog_features(feature_image[:,:,channel],
                                    vis=False, feature_vec=True)
                hog_features.append(channel_hog_feature)
            hog_features = np.ravel(hog_features)
            
        else:
            hog_features = hog.get_hog_features(feature_image[:,:,hog_channel], orient, 
                        vis=False, feature_vec=True)
        features.append(hog_features)

    return features

def extract_features(imgs, cspace='RGB', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, 
spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256), bin_spatial_feat=True, color_hist_feat=True, hog_feat=True):
    features = []
    hog = HOG(orient, pix_per_cell, cell_per_block)

    for imgSrc in imgs:
        feature = []
        img = mpimg.imread(imgSrc)
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(img)

        if bin_spatial_feat == True:
            spatial_features = bin_spatial(feature_image, cspace, spatial_size)
            feature.append(spatial_features)
        
        if color_hist_feat == True:
            hist_features = color_hist(feature_image, hist_bins, hist_range)
            feature.append(hist_features)

        if hog_feat == True:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    channel_hog_feature = hog.get_hog_features(feature_image[:,:,channel],
                                        vis=False, feature_vec=True)
                    hog_features.append(channel_hog_feature)
                hog_features = np.ravel(hog_features)
                
            else:
                hog_features = hog.get_hog_features(feature_image[:,:,hog_channel], orient,
                            vis=False, feature_vec=True)
            feature.append(hog_features)
        features.append(np.concatenate(feature))
    return features

def extract_features_single_img(img, cspace='RGB', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, 
spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256), bin_spatial_feat=True, color_hist_feat=True, hog_feat=True):
    hog = HOG(orient, pix_per_cell, cell_per_block)
    feature = []
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)

    if bin_spatial_feat == True:
        spatial_features = bin_spatial(feature_image, cspace, spatial_size)
        feature.append(spatial_features)
    
    if color_hist_feat == True:
        hist_features = color_hist(feature_image, hist_bins, hist_range)
        feature.append(hist_features)

    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                channel_hog_feature = hog.get_hog_features(feature_image[:,:,channel],
                                    vis=False, feature_vec=True)
                hog_features.append(channel_hog_feature)
            hog_features = np.ravel(hog_features)
            
        else:
            hog_features = hog.get_hog_features(feature_image[:,:,hog_channel], orient, 
                        vis=False, feature_vec=True)
        feature.append(hog_features)

    return np.concatenate(feature)
