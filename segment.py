import cv2
import numpy as np
from typing import List
from utils import visualize_resize


def segment_chars(img: np.ndarray, visualize: bool = False, easy: bool = False) -> List[np.ndarray]:
    """Extracts the characters from the license plate and returns a list of the descriptors of them.

    Args:
        img (np.ndarray): The license plate image.
        visualize (bool, optional): Whether to visualize the segmentation process. Defaults to False.

    Returns:
        List[np.ndarray]: Characters in the image.
    """

    # Apply median blur to smoothen the image
    blur = cv2.medianBlur(img, 5)
    if visualize:
        visualize_resize(blur, "blur", height=500)

    # Binarize the image
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Process the green plate further
    half = thresh.shape[0]*thresh.shape[1]*255/2
    is_green = False
    if thresh.sum() > half:
        is_green = True
        # Extract the black characters
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        thresh = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 70]))
        if visualize:
            visualize_resize(thresh, "char_mask", height=500)

    else:
        # Removes the white border
        # if visualize:
        #     visualize_resize(thresh, 'before', close=False)
        # Remove the horizontal white borders
        r_thresh = thresh.shape[1] * 9/10
        for i in range(thresh.shape[0]//2, -1, -1):
            if (thresh[i] == 0).sum() > r_thresh:
                thresh[:i,:] = 0
                break
        for i in range(thresh.shape[0]//2, thresh.shape[0]):
            if (thresh[i] == 0).sum() > r_thresh:
                thresh[i:,:] = 0
                break
        # Remove the vertical white borders
        c_thresh = thresh.shape[0] * 3/4
        for j in range(thresh.shape[1]):
            if (thresh[:,j] == 0).sum() < c_thresh:
                thresh[:,:j] = 0
                break
        for j in range(thresh.shape[1]-1, -1, -1):
            if (thresh[:,j] == 0).sum() < c_thresh:
                thresh[:,j:] = 0
                break
        # if visualize:
            # visualize_resize(thresh, 'after')

    # Erode and then dilate to make the characters more clear
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    eroded = cv2.erode(thresh, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=20 if easy else 5 if is_green else 8)

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
        visualize_resize(demo_tmp, 'borders')

    # Extract the regions of interest
    rois = [cv2.resize(thresh[y:y+h, x:x+w], (256,512), interpolation=cv2.INTER_NEAREST)
             for x, y, w, h in borders if h > thresh.shape[0]*0.5 and w*1.2 < h < w*4]
    
    # Further process
    # Remove noise by median blur
    res_rois = [cv2.medianBlur(roi, 23) for roi in rois]
    # Remove the white border of letters and numbers
    for idx in range(1, len(res_rois)):
        conts, _ = cv2.findContours(res_rois[idx], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(cont) for cont in conts]
        if len(boxes) > 1:
            boxes.sort(key=lambda x: x[-1]*x[-2])
            # Keep the largest box only
            for box in boxes[:-1]:
                x, y, w, h = box
                res_rois[idx][y:y+h, x:x+w] = 0
    # Remove the black padding
    for idx in range(len(res_rois)):
        for i in range(res_rois[idx].shape[0]):
            if res_rois[idx][i].sum():
                res_rois[idx] = res_rois[idx][i:]
                break
        for i in range(res_rois[idx].shape[0]-1, -1, -1):
            if res_rois[idx][i].sum():
                res_rois[idx] = res_rois[idx][:i+1]
                break
        for j in range(res_rois[idx].shape[1]):
            if res_rois[idx][:,j].sum():
                res_rois[idx] = res_rois[idx][:,j:]
                break
        for j in range(res_rois[idx].shape[1]-1, -1, -1):
            if res_rois[idx][:,j].sum():
                res_rois[idx] = res_rois[idx][:,:j+1]
                break

    # Visualize the segmented characters
    if visualize:
        for roi, res_roi in zip(rois, res_rois):
            # Compare the original roi and the processed roi
            visualize_resize(roi, 'roi', height=400, close=False)
            visualize_resize(res_roi, 'res_roi', height=400, close=True)
        
    return res_rois
