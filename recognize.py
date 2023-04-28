import numpy as np
import cv2
from sklearn.decomposition import PCA


def get_HOG_descriptor(img: np.ndarray) -> np.ndarray:
    """Computes the HOG descriptor of the image.

    Args:
        img (np.ndarray): The input image.

    Returns:
        np.ndarray: The HOG descriptor of the image.
    """    

    hog = cv2.HOGDescriptor(_winSize=(256,512), _blockSize=(64,64), _blockStride=(32,32), _cellSize=(32,32), _nbins=9)
    return hog.compute(img)


def recognize_char(char: np.ndarray) -> str:
    """Recognizes the character.

    Args:
        char (np.ndarray): The input character image.

    Returns:
        str: The character.
    """

    descriptor = get_HOG_descriptor(char)

