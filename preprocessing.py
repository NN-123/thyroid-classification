from skimage import io
from skimage.filters import gaussian
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage.transform import resize
import os
import numpy as np

path_to_data = os.path.join('.', 'data')
path_to_jpg = os.path.join(path_to_data, 'jpg')
path_to_preprocessed = os.path.join(path_to_data, 'preprocessed')
filenames = os.listdir(path_to_jpg)

image_height = 360
image_width = 560
binary_thresh = 3
gaussian_sigma = 5
slicing = (20, 25, -30, -25)


def preprocess(image):
    image = image[20:-30, 25:-25, :]
    assert isinstance(image, np.ndarray)
    grayscale = rgb2gray(image)
    grayscale = gaussian(grayscale, sigma=gaussian_sigma)

    binary = grayscale * 255 > binary_thresh

    labelled = label(binary)
    regions = regionprops(labelled)
    areas = [region.area for region in regions]
    arg = int(np.argmax(areas))
    h1, w1, h2, w2 = regions[arg].bbox
    extracted = image[h1: h2, w1: w2, :]
    return (resize(extracted, (image_height, image_width)) * 255).astype(np.uint8)


for filename in filenames:
    im = io.imread(os.path.join(path_to_jpg, filename), as_gray=False)
    processed = preprocess(im)
    io.imsave(os.path.join(path_to_preprocessed, filename), processed)
    print('Successfully processed %s' % filename)
