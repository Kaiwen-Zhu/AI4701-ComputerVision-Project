import numpy as np
import cv2
import os
from typing import List
from sklearn.metrics.pairwise import cosine_similarity


def get_HOG_descriptor(img: np.ndarray) -> np.ndarray:
    """Computes the HOG descriptor of the image.

    Args:
        img (np.ndarray): The input image.

    Returns:
        np.ndarray: The HOG descriptor of the image.
    """    

    img = cv2.resize(img, (48,96), interpolation=cv2.INTER_NEAREST)
    img = cv2.copyMakeBorder(img, 8,8,4,4, cv2.BORDER_CONSTANT, value=(0,0,0))
    hog = cv2.HOGDescriptor(_winSize=(56,112), _blockSize=(14,14), _blockStride=(7,7), _cellSize=(7,7), _nbins=9)
    return hog.compute(img)


def KNN_recognize(train_data: dict, target: np.ndarray, K: int = 7) -> str:
    """Recognizes the character using K-Nearest Neighbors algorithm.

    Args:
        train_data (dict): Training data.
        target (np.ndarray): The HOG descriptor of the character to be recognized.
        K (int): The number of neighbors to consider.

    Returns:
        str: The character.
    """
    
    # Compute the distance between the target and each training data
    distances = []
    for label in train_data.keys():
        for sample in train_data[label]:
            distances.append((label, cosine_similarity(target.reshape(1,-1), sample.reshape(1,-1))))

    # Find the K nearest neighbors
    neighbors = sorted(distances, reverse=True, key=lambda x: x[1])[:K]

    # Find the label with the most neighbors
    labels = [neighbor[0] for neighbor in neighbors]
    label = max(set(labels), key=labels.count)

    return label


def recognize_char(char: np.ndarray) -> str:
    """Recognizes the character.

    Args:
        char (np.ndarray): The input character image.

    Returns:
        str: The character.
    """

    # Load data and compute the HOG descriptors
    labels = os.listdir("training_data")
    filenames = {label: os.listdir(os.path.join("training_data", label)) for label in labels}
    train_data = {}
    for label, fs in filenames.items():
        train_data[label] = []
        for f in fs:
            img = cv2.imdecode(np.fromfile(os.path.join("training_data", label, f), dtype=np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            train_data[label].append(get_HOG_descriptor(img))

    # Compute the HOG descriptor of the character to be recognized
    target = get_HOG_descriptor(char)

    # Use K-Nearest Neighbors algorithm to recognize the character
    res = KNN_recognize(train_data, target)

    return res


def recognize_licenses(licenses: List[List]) -> None:
    """Recognizes the licenses and writes the results into the parameter `licenses`.

    Args:
        licenses (List[List]): A list of pairs of filenames and license plate images (np.ndarray).
    """    

    # Load data and compute the HOG descriptors
    labels = os.listdir("training_data")
    filenames = {label: os.listdir(os.path.join("training_data", label)) for label in labels}
    train_data = {}
    for label, fs in filenames.items():
        train_data[label] = []
        for f in fs:
            img = cv2.imdecode(np.fromfile(os.path.join("training_data", label, f), dtype=np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            train_data[label].append(get_HOG_descriptor(img))

    # Recognize characters in each license plate
    for i in range(len(licenses)):
        res = ""
        for license in licenses[i][1]:
            # Compute the HOG descriptor of the character to be recognized
            target = get_HOG_descriptor(license)
            # Use K-Nearest Neighbors algorithm to recognize the character
            res += KNN_recognize(train_data, target)
        # Write the result
        licenses[i][1] = res
    licenses.sort()
