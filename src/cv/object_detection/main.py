import cv2

from find_vehicle.main import find_vehicles
from train import train, load_train_data
from draw.main import draw_boxes
from scipy.ndimage.measurements import label

test_img = cv2.imread('../../../test_images/test1.jpg')
ystart = 400
ystop = 656
scale = 1.5
colorspace = 'YUV'
orient = 11
pix_per_cell = 16
cell_per_block = 2
hog_channel = "ALL"
spatial_size = (32, 32)
hist_bins = 32
spatial_feat = True
hist_feat = True
hog_feat = True


svc, X_scaler = train(colorspace, orient, pix_per_cell, cell_per_block, hog_channel, False)
rectangles = []

colorspace = 'YUV'
orient = 11
pix_per_cell = 16
cell_per_block = 2
hog_channel = 'ALL'


ystart = 400
ystop = 464
scale = 1.0
rectangles.append(find_vehicles(test_img, ystart, ystop, scale, colorspace, hog_channel, svc, X_scaler,
                       orient, pix_per_cell, cell_per_block, None, None))
ystart = 416
ystop = 480
scale = 1.0
rectangles.append(find_vehicles(test_img, ystart, ystop, scale, colorspace, hog_channel, svc, X_scaler,
                       orient, pix_per_cell, cell_per_block, None, None))
ystart = 400
ystop = 496
scale = 1.5
rectangles.append(find_vehicles(test_img, ystart, ystop, scale, colorspace, hog_channel, svc, X_scaler,
                       orient, pix_per_cell, cell_per_block, None, None))
ystart = 432
ystop = 528
scale = 1.5
rectangles.append(find_vehicles(test_img, ystart, ystop, scale, colorspace, hog_channel, svc, X_scaler,
                       orient, pix_per_cell, cell_per_block, None, None))
ystart = 400
ystop = 528
scale = 2.0
rectangles.append(find_vehicles(test_img, ystart, ystop, scale, colorspace, hog_channel, svc, X_scaler,
                       orient, pix_per_cell, cell_per_block, None, None))
ystart = 432
ystop = 560
scale = 2.0
rectangles.append(find_vehicles(test_img, ystart, ystop, scale, colorspace, hog_channel, svc, X_scaler,
                       orient, pix_per_cell, cell_per_block, None, None))
ystart = 400
ystop = 596
scale = 3.5
rectangles.append(find_vehicles(test_img, ystart, ystop, scale, colorspace, hog_channel, svc, X_scaler,
                       orient, pix_per_cell, cell_per_block, None, None))
ystart = 464
ystop = 660
scale = 3.5
rectangles.append(find_vehicles(test_img, ystart, ystop, scale, colorspace, hog_channel, svc, X_scaler,
                       orient, pix_per_cell, cell_per_block, None, None))

rectangles = [item for sublist in rectangles for item in sublist]
test_img_rects = draw_boxes(test_img, rectangles, color='random', thick=2)
cv2.imshow('draw', test_img_rects)
cv2.waitKey(0)
cv2.destroyAllWindows()