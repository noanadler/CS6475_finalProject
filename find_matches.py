
import numpy as np
import cv2

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

def find_top_matches_distances(image_1, image_2, top_matches_num):
    """

    This function detects and computes SIFT (or ORB) from the input images, and
    returns the best matches using the normalized Hamming Distance.

    Args:
      image_1 (numpy.ndarray): The first image (grayscale).
      image_2 (numpy.ndarray): The second image. (grayscale).
      top_matches_num : number of top mathces

    Returns:
      top_distances array that contains the distances between the kp of the top matches

    """
    orb = cv2.ORB()

    #convert images to greyscale (not sure this is needed)
    image_1 = cv2.cvtColor(image_1,cv2.COLOR_BGR2GRAY)
    image_2 = cv2.cvtColor(image_2,cv2.COLOR_BGR2GRAY)

    image_1_kp, image_1_desc = orb.detectAndCompute(image_1, None)
    image_2_kp, image_2_desc = orb.detectAndCompute(image_2, None)
    # Create BFMatcher (Brute Force Matcher) object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors
    matches = bf.match(image_1_desc, image_2_desc)
    # Sort them in the order of their distance
    matches = sorted(matches, key = lambda x: x.distance)
    # Get first 10 matches
    matches_top = matches[:top_matches_num]
    # Create an array of top distances
    top_distances = np.zeros(len(matches_top))
    for i in range(0, len(top_distances)):
        top_distances[i] = matches[i].distance

    return top_distances