import numpy as np
import cv2
import find_matches
import os
from glob import glob


key_points_num = 15

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

def find_sift_best_match(images, test_img):
    """This function finds the distances of the top matches between the test_img and each image
    from the images list"
    It returns the name of the image that has the closest matches with test_img.
    """
    # top_distances_sum is a dictionary, the key is the name of the img and the value
    # is the sum of the top matches distances
    top_distances_sum = {}

    for key in images.keys():
        top_distances = find_matches.find_top_matches_distances(test_img, images[key], key_points_num)
        top_distances_sum[key] = top_distances.sum()
    # Find the flower with the min sum of distances
    result = min(top_distances_sum, key=top_distances_sum.get)

    return result


def find_color_histogram_best_match(images, test_img):
    # used this as referance: http://www.pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/
    # index stores the histograms
    histograms_dictionary = {}
    ranking = {}

    # Get color histogram of test_img
    # Create mask
    grey = cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    mask = np.zeros(grey.shape[:2], np.uint8)
    mask[grey != grey[0, 0]] = 255
    # Convert to hsv
    img_hsv = cv2.cvtColor(test_img, cv2.COLOR_BGR2HSV)
    # Get normalized histogram
    test_hist = cv2.calcHist([img_hsv], [0, 1], mask, [180, 256], [0, 180, 0, 256])
    norm_test_hist = cv2.normalize(test_hist).flatten()

    # Get color histogram of db images
    for key in images.keys():
        # Initialize ranking dictionary
        ranking[key] = 0
        # Create mask
        grey = cv2.cvtColor(images[key],cv2.COLOR_BGR2GRAY)
        mask = np.zeros(grey.shape[:2], np.uint8)
        mask[grey != grey[0, 0]] = 255
        # Convert to hsv
        img_hsv = cv2.cvtColor(images[key], cv2.COLOR_BGR2HSV)
        # Get normalized histogram
        img_hist = cv2.calcHist([img_hsv], [0, 1], mask, [180, 256], [0, 180, 0, 256])
        norm_img_hist = cv2.normalize(img_hist).flatten()
        histograms_dictionary[key] = norm_img_hist

    # initialize OpenCV methods for histogram comparison
    OPENCV_METHODS = (
        ("Correlation", cv2.cv.CV_COMP_CORREL),
        ("Chi-Squared", cv2.cv.CV_COMP_CHISQR),
        ("Intersection", cv2.cv.CV_COMP_INTERSECT),
        ("Hellinger", cv2.cv.CV_COMP_BHATTACHARYYA))

    # loop over the comparison methods
    for (methodName, method) in OPENCV_METHODS:
        # initialize the results dictionary and the sort direction
        results = {}
        reverse = False
        # if we are using the correlation or intersection method, then sort the results in reverse order
        if methodName in ("Correlation", "Intersection"):
            reverse = True

        for (key, hist) in histograms_dictionary.items():
            # compute the distance between the two histograms
            # using the method and update the results dictionary
            d = cv2.compareHist(norm_test_hist, hist, method)
            results[key] = d
        # sort the results
        results = sorted([(v, k) for (k, v) in results.items()], reverse=reverse)
        score, winner = results[0]

        # ranking the results
        ranking[winner] += 1
    result = max(ranking, key=ranking.get)
    return result

def find_gradients_histogram_best_match(images, test_img):
    hog_dictionary = {}
    ranking = {}

    # Find HOG of test_img
    hog = cv2.HOGDescriptor()
    test_hist = hog.compute(test_img)
    norm_test_hist = cv2.normalize(test_hist).flatten()

    # Find HOG for db images
    for key in images.keys():
        # Initialize ranking dictionary
        ranking[key] = 0
        img_hog = hog.compute(images[key])
        norm_img_hog = cv2.normalize(img_hog).flatten()
        hog_dictionary[key] = norm_img_hog

        # initialize OpenCV methods for histogram comparison
    OPENCV_METHODS = (
        ("Correlation", cv2.cv.CV_COMP_CORREL),
        ("Chi-Squared", cv2.cv.CV_COMP_CHISQR),
        ("Intersection", cv2.cv.CV_COMP_INTERSECT),
        ("Hellinger", cv2.cv.CV_COMP_BHATTACHARYYA))

    # loop over the comparison methods
    for (methodName, method) in OPENCV_METHODS:
        # initialize the results dictionary and the sort direction
        results = {}
        reverse = False
        # if we are using the correlation or intersection method, then sort the results in reverse order
        if methodName in ("Correlation", "Intersection"):
            reverse = True

        for (key, hist) in hog_dictionary.items():
            # compute the distance between the two histograms
            # using the method and update the results dictionary
            d = cv2.compareHist(norm_test_hist, hist, method)
            results[key] = d
        # sort the results
        results = sorted([(v, k) for (k, v) in results.items()], reverse=reverse)
        score, winner = results[0]

        # ranking the results
        ranking[winner] += 1
    result = max(ranking, key=ranking.get)
    return result