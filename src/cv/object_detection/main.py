import cv2

from find_vehicle.main import find_vehicles
from train import train

test_img = cv2.imread('C:\\Users\ZHE WANG\source\object-perception-self-driving\\test_images\\test1.jpg')
cv2.imshow('image', test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

ystart = 400
ystop = 656
scale = 1.5
colorspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11
pix_per_cell = 16
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"


svc = train()

rectangles = find_vehicles(test_img, ystart, ystop, scale, colorspace, hog_channel, svc, None, orient, pix_per_cell, cell_per_block, None, None)

print(len(rectangles), 'rectangles found in image')