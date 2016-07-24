import numpy as np
import cv2
import find_matches

# Import ORB as SIFT to avoid confusion.
try:
    from cv2 import ORB as SIFT
except ImportError:
    try:
        from cv2 import SIFT
    except ImportError:
        try:
            SIFT = cv2.ORB_create
        except:
            raise AttributeError("Version of OpenCV(%s) does not have SIFT / ORB."
                                 % cv2.__version__)

img_width = 245
img_hight = 205
key_points_num = 15
test_key = 'test'

def readImages(image_dir):
    """ This function reads in input images from a image directory

    Args:
        image_dir (str): The image directory to get images from.

    Returns:
        images(dictionary): List of images in image_dir. Each image in the list is of
                      type numpy.ndarray.
        flower_names(list) : List of the names of the flowers in the db

    """
    images = {}
    extensions = ['bmp', 'pbm', 'pgm', 'ppm', 'sr', 'ras', 'jpeg',
                  'jpg', 'jpe', 'jp2', 'tiff', 'tif', 'png']

    search_paths = [os.path.join(image_dir, '*.' + ext) for ext in extensions]
    image_files = sorted(reduce(list.__add__, map(glob, search_paths)))
    for f in image_files:
        images[f[f.rfind("/") + 1:f.rfind(".")]] = cv2.imread(f, cv2.IMREAD_UNCHANGED | cv2.IMREAD_COLOR)

    return images

def remove_background(images):

    for key in images.keys():
        images[key][(images[key] > 220).all(axis=2)] = [225, 0, 0]
    return images

def rezise_images(images):

    for key in images.keys():
        images[key] = cv2.resize(images[key], (img_width, img_hight))
    return images
