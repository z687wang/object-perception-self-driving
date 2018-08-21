import cv2

from find_vehicle.main import find_vehicles
from train import train

test_img = cv2.imread('../../../test_images/test1.jpg')
cv2.imshow('image', test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 15  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off


svc = train(color_space, orient, pix_per_cell, cell_per_block, hog_channel)
rectangles = find_cars(test_img, ystart, ystop, scale, colorspace, hog_channel, svc, None, orient, pix_per_cell, cell_per_block, None, None)
print(len(rectangles), 'rectangles found in image')