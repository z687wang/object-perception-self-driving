import cv2
import numpy as np
import sys
sys.path.append('../')
from feature_extraction.main import extract_hog_features
from hog.main import HOG
from color_hist.main import color_hist
from bin_spatial.main import bin_spatial
from hog.main import HOG

def find_vehicles(img, ystart, ystop, scale, cspace, hog_channel, svc, X_scaler, orient, 
              pix_per_cell, cell_per_block, spatial_size=(32, 32), hist_bins=32, show_all_rectangles=False):

    rectangles = []
    hog = HOG(orient, pix_per_cell, cell_per_block)
    img = img.astype(np.float32)/255
    
    img_cropped = img[ystart:ystop,:,:]

    if cspace != 'RGB':
        if cspace == 'HSV':
            img_to_search = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            img_to_search = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            img_to_search = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            img_to_search = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            img_to_search = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2YCrCb)
    else: img_to_search = np.copy(img_cropped)
    
    # rescale image if other than 1.0 scale
    if scale != 1:
        imshape = img_to_search.shape
        img_to_search = cv2.resize(img_to_search, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    # select colorspace channel for HOG 
    if hog_channel == 'ALL':
        ch1 = img_to_search[:,:,0]
        ch2 = img_to_search[:,:,1]
        ch3 = img_to_search[:,:,2]
    else: 
        ch1 = img_to_search[:,:,hog_channel]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)+1  #-1
    nyblocks = (ch1.shape[0] // pix_per_cell)+1  #-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = hog.get_hog_features(ch1, False, feature_vec=False)   
    if hog_channel == 'ALL':
        hog2 = hog.get_hog_features(ch2, False, feature_vec=False)
        hog3 = hog.get_hog_features(ch3, False, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            if hog_channel == 'ALL':
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_features = hog_feat1

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            spatial_features = bin_spatial(img_to_search)
            hist_features, _, _, _, _ = color_hist(img_to_search)
            test_features = X_scaler.transform(np.hstack((spatial_features, hog_features)).reshape(1, -1))
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1 or show_all_rectangles:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                rectangles.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
    return rectangles