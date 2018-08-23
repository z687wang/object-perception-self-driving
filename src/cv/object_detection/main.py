import cv2
import matplotlib.image as mpimg
from find_vehicle.main import find_vehicles, slide_window, search_windows
from train import train, load_train_data
from draw.main import draw_boxes
from scipy.ndimage.measurements import label

test_img = cv2.imread('../../../test_images/test4.jpg')

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


svc, X_scaler = train(colorspace, orient, pix_per_cell, cell_per_block, hog_channel, False, bin_spatial_feat, color_hist_feat, hog_feat)
rectangles = []
windows =  slide_window(test_img,
                        x_start_stop=[600, None],
                        y_start_stop=[400, 656],
                        xy_window=(64,64),
                        xy_overlap=(.7,.7))
detect_windows = search_windows(test_img, windows, svc, X_scaler, color_space=colorspace, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, bin_spatial_feat=bin_spatial_feat, 
                            color_hist_feat=color_hist_feat, hog_feat=hog_feat)  
test_img_rects = draw_boxes(test_img, detect_windows, 'random', 2)
cv2.imshow('draw', test_img_rects)
cv2.waitKey(0)
cv2.destroyAllWindows()