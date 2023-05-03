import cv2
import numpy as np
from utils import visualize_resize


def rectify_plate(img: np.ndarray, corners: np.ndarray, visualize: bool = False) -> np.ndarray:
    """Rectifies the slant license plate image by perspective transformation.

    Args:
        img (np.ndarray): The slant license plate image.
        corners (np.ndarray): The corners of the plate.
        visualize (bool, optional): Whether to visualize the rectification process. Defaults to False.

    Returns:
        np.ndarray: The rectified license plate image.
    """    

    # Sort the corners so that the order is top-left, bottom-left, bottom-right, top-right
    corners = np.array(sorted(corners, key=lambda x: x[0]))
    if corners[0][1] > corners[1][1]:
        tmp = corners[0].copy()
        corners[0] = corners[1]
        corners[1] = tmp
    if corners[2][1] < corners[3][1]:
        tmp = corners[2].copy()
        corners[2] = corners[3]
        corners[3] = tmp

    # Get the width and height
    dx = corners[-1][0] - corners[0][0]
    dy = corners[1][1] - corners[0][1]

    # Compute the destination points
    dst = np.float32([[0,0], [0, dy], [dx, dy], [dx, 0]])
    src = np.float32(corners)

    # Compute the transformation matrix
    T = cv2.getPerspectiveTransform(src, dst)
    # Apply the transformation
    rectified = cv2.warpPerspective(img, T, (dx, dy))

    if visualize:
        visualize_resize(rectified, "rectified", height=500)

    return rectified
