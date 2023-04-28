import cv2
import numpy as np
from utils import visualize_resize


def segment_chars(img: np.ndarray, visualize: bool = False) -> list[np.ndarray]:
    """Extracts the characters from the license plate and returns a list of the descriptors of them.

    Args:
        img (np.ndarray): The license plate image.
        visualize (bool, optional): Whether to visualize the segmentation process. Defaults to False.

    Returns:
        list[np.ndarray]: Characters in the image.
    """

    # Binarize the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Invert the image if the text is black to ensure the text is white and the background is black
    half = thresh.shape[0]*thresh.shape[1]*255/2
    if thresh.sum() > half:
        thresh = cv2.bitwise_not(thresh)

    # Erode and then dilate to make the characters more clear
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    eroded = cv2.erode(thresh, kernel, iterations=5)
    dilated = cv2.dilate(eroded, kernel, iterations=15)

    # Visualize the eroded and dilated image
    if visualize:
        visualize_resize(thresh, 'thresh', close=False)
        visualize_resize(eroded, 'eroded', close=False)
        visualize_resize(dilated, 'dilated', close=True)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # each element in borders is (x, y, w, h) of the bounding box of a character
    borders = [cv2.boundingRect(contour) for contour in contours]
    borders.sort(key=lambda x: x[0])

    # Draw the borders on the image
    if visualize:
        demo_tmp = cv2.cvtColor(dilated, cv2.COLOR_GRAY2RGB)
        for border in borders:
            x, y, w, h = border
            cv2.rectangle(demo_tmp, (x, y), (x+w, y+h), (0,0,255), 3)
        visualize_resize(demo_tmp, 'demo_tmp')

    # Extract the regions of interest
    rois = [thresh[y:y+h, x:x+w] for x, y, w, h in borders if h > thresh.shape[0]*0.7]
    # Pad the rois
    rois = [cv2.resize(cv2.copyMakeBorder(roi, 10,10,10,10, cv2.BORDER_CONSTANT, value=(0,0,0)),
                        (256,512), interpolation=cv2.INTER_NEAREST) for roi in rois]
    # Remove noise by median blur
    res_rois = [cv2.medianBlur(roi, 23) for roi in rois]

    # Visualize the segmented characters
    if visualize:
        for roi, res_roi in zip(rois, res_rois):
            # Compare the original roi and the processed roi
            visualize_resize(roi, 'roi', height=400, close=False)
            visualize_resize(res_roi, 'res_roi', height=400, close=True)

    return res_rois
