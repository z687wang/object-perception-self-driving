from skimage.feature import hog
import cv2


pix_per_cell = 8
cell_per_block = 2
orient = 9

class HOG:
    def __init__(self, orient=9, pix_per_cell=8, cell_per_block=2):
        self.orientations = orient
        self.pix_per_cell = pix_per_cell
        self.pix_per_block = cell_per_block

    def get_hog_features(self, img, vis=False, feature_vec=False):
        return_list = hog(img, orientations=self.orientations, pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  block_norm= 'L2-Hys', transform_sqrt=False,
                                  visualize= vis, feature_vector= feature_vec)
        hog_features = return_list[0]
        if vis:
            hog_img = return_list[1]
            return hog_features, hog_img
        else:
            return hog_features


if __name__ == '__main__':
    imgSrc = '../../../../test_images/test1.jpg'
    img = cv2.imread(imgSrc)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Hog = HOG(9, 8, 2)
    hog_features, hog_img = Hog.get_hog_features(gray)
    cv2.imshow('HOG', hog_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
