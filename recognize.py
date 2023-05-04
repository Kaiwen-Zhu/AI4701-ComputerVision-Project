import numpy as np
import cv2
import os
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import manifold
import matplotlib.pyplot as plt
from matplotlib import cm
from evaluate import truth


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


def visualize_tSNE(train_data: dict, targets: List[np.ndarray], truth: str) -> None:
    """Visualizes data using t-SNE.

    Args:
        train_data (dict): Training data.
        targets (List[np.ndarray]): Test data.
        truth (str): The ground truth.
    """    

    # Compute the t-SNE embedding
    labels = list(train_data.keys())
    n_lbl = len(labels)
    X = []
    for label in labels:
        X.extend(train_data[label])
    X.extend(targets)
    X = np.array(X)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=1453, learning_rate='auto')
    X_embedded = tsne.fit_transform(X)
    embeddings = {labels[i]: X_embedded[i*7:(i+1)*7] for i in range(n_lbl)}

    # Visualize the embedding
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    cmap = cm.rainbow(np.linspace(0, 1, n_lbl))
    # Visualize the training data
    for i in range(n_lbl):
        label = labels[i]
        if '0' <= label <= '9':
            marker = 'x'
        elif 'A' <= label <= 'Z':
            marker = '+'
        else:
            marker = '^'
        embedding = embeddings[label]
        plt.scatter(embedding[:,0], embedding[:,1], label=label, 
                    color=cmap[i], marker=marker, s=10, linewidths=0.5)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5,1), borderaxespad=1, ncol=10)
    # Visualize the test data
    for i in range(len(targets)):
        x, y = X_embedded[7*n_lbl+i,0], X_embedded[7*n_lbl+i,1]
        plt.scatter(x, y, s=2, color='black')
        plt.annotate(truth[i], xy = (x,y), xytext = (x+0.1, y+0.1))

    # plt.savefig("visualization_res/tsne.svg", format='svg', bbox_inches='tight', pad_inches=0)
    plt.savefig("visualization_res/tsne.png", dpi=1000, bbox_inches='tight', pad_inches=0)
    plt.show()


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


def recognize_licenses(licenses: List[List], visualize: bool = False) -> None:
    """Recognizes the licenses and writes the results into the parameter `licenses`.

    Args:
        licenses (List[List]): A list of pairs of filenames and license plate images (np.ndarray).
        visualize (bool, optional): Whether to visualize the recognition process. Defaults to False.
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
        targets = []
        for license in licenses[i][1]:
            # Compute the HOG descriptor of the character to be recognized
            target = get_HOG_descriptor(license)
            targets.append(target)
            # Use K-Nearest Neighbors algorithm to recognize the character
            res += KNN_recognize(train_data, target)
        # Write the result
        licenses[i][1] = res
        # Visualize the t-SNE embedding
        if visualize:
            visualize_tSNE(train_data, targets, truth[licenses[i][0]])
    licenses.sort()
