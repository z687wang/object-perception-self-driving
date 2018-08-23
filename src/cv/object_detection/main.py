import cv2
import numpy as np
import matplotlib.image as mpimg
from find_vehicle.main import find_vehicles, slide_window, search_windows
from find_vehicle.process import detect_vehicles
from train import train, load_train_data
from draw.main import draw_boxes, draw_labeled_bboxes


test_img = cv2.imread('../../../test_images/test6.jpg')
test_img = test_img.astype(np.float32)/255
colorspace = 'YUV'
orient = 15
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL"
spatial_size = (32, 32)
hist_bins = 32
bin_spatial_feat = True
color_hist_feat = True
hog_feat = True
load_model = True

test_img_rects = detect_vehicles(test_img, colorspace, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, bin_spatial_feat, color_hist_feat, hog_feat, load_model)

cv2.imshow('draw', test_img_rects)
cv2.waitKey(0)
cv2.destroyAllWindows()
