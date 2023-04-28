import cv2
import numpy as np


def visualize_resize(src: np.ndarray, title: str, height: int = 200, close: bool = True) -> None:
    """Visualizes the image with the given height.

    Args:
        src (np.ndarray): The image to visualize.
        title (str): The title of the window.
        height (int, optional): The desired height of the window. Defaults to 200.
        close (bool, optional): Whether to close the window on key press. Defaults to True.
    """    

    cv2.namedWindow(title, 0)
    cv2.resizeWindow(title, int(height*src.shape[1]/src.shape[0]), height)
    cv2.imshow(title, src)
    if close:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
