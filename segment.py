import cv2
import numpy as np


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
        height = 200
        def visualize_resize(src, title):
            cv2.namedWindow(title, 0)
            cv2.resizeWindow(title, int(height*src.shape[1]/src.shape[0]), height)
            cv2.imshow(title, src)
        visualize_resize(thresh, 'thresh')
        visualize_resize(eroded, 'eroded')
        visualize_resize(dilated, 'dilated')
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Extract the regions of interest
    rois = [thresh[y:y+h, x:x+w] for x, y, w, h in borders if h > thresh.shape[0]*0.7]
    # Pad the rois
    rois = [cv2.resize(cv2.copyMakeBorder(roi, 10,10,10,10, cv2.BORDER_CONSTANT, value=(0,0,0)),
                        (256,512), interpolation=cv2.INTER_NEAREST) for roi in rois]
    # rois = [cv2.copyMakeBorder(roi, 10,10,10,10, cv2.BORDER_CONSTANT, value=(0,0,0)) for roi in rois]
    # Erode and then dilate to remove the noise
    e_rois = [cv2.erode(roi, kernel, iterations=2) for roi in rois]
    d_rois = [cv2.dilate(e_roi, kernel, iterations=4) for e_roi in e_rois]
    # Smoothen the edges
    b_rois = [cv2.GaussianBlur(d_roi, (51,51), 0) for d_roi in d_rois]
    res_rois = [cv2.threshold(b_roi, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] for b_roi in b_rois]

    # Visualize the segmented characters
    if visualize:
        for roi, e_roi, d_roi, b_roi, res_roi in zip(rois, e_rois, d_rois, b_rois, res_rois):
            # Compare the original roi and the processed roi
            cv2.imshow('roi', roi)
            cv2.imshow('e_roi', e_roi)
            cv2.imshow('d_roi', d_roi)
            cv2.imshow('b_roi', b_roi)
            cv2.imshow('res_roi', res_roi)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return res_rois
