import cv2
import numpy as np
from typing import Tuple
from utils import visualize_resize


def locate_plate(img: np.ndarray, visualize: bool = False) -> Tuple[np.ndarray]:
    """Locates the license plate in the image.

    Args:
        img (np.ndarray): The input image.
        visualize (bool, optional): Whether to visualize the localization process. Defaults to False.

    Returns:
        Tuple[np.ndarray]: The license plate image and the corners.
    """    

    # Apply median blur to smoothen the image
    blur = cv2.medianBlur(img, 51)
    if visualize:
        visualize_resize(img, "img", height=500, close=False)
        visualize_resize(blur, "blur", height=500)

    # Convert to HSV to filter out the blue and green regions
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, np.array([100, 110, 110]), np.array([110, 255, 255]))
    green_mask = cv2.inRange(hsv, np.array([35, 0, 180]), np.array([70, 255, 255]))
    plate_mask = cv2.bitwise_or(blue_mask, green_mask)
    if visualize:
        visualize_resize(plate_mask, "plate_mask", height=500, close=False)

    # Dilate to make each object more connected
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dilated = cv2.dilate(plate_mask, kernel, iterations=8)
    if visualize:
        visualize_resize(dilated, "dilated", height=500)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if visualize:
        demo_img = img.copy()
        cv2.drawContours(demo_img, contours, -1, (0,0,255), 5)
        visualize_resize(demo_img, "contours", height=500)

    # Filter for quadrangles
    corners = []
    for contour in contours:
        polygon = cv2.approxPolyDP(contour, 0.05*cv2.arcLength(contour, True), True)
        if len(polygon) == 4:
            corners.append(polygon)

    if visualize:
        demo_img = img.copy()
        cv2.drawContours(demo_img, corners, -1, (0,0,255), 5)
        visualize_resize(demo_img, "corners", height=500)

    # Cut the license plate from the image
    corners = max(corners, key=cv2.contourArea)
    minx = min(corners[:,0,0])
    miny = min(corners[:,0,1])
    maxx = max(corners[:,0,0])
    maxy = max(corners[:,0,1])
    roi = img[miny:maxy, minx:maxx]

    if visualize:
        visualize_resize(roi, "roi", height=500)

    return roi, np.array([[corner[0]-minx, corner[1]-miny] for corner in corners.reshape(4,2)])


if __name__ == "__main__":
    path = "resources/images/medium/2-3.jpg"
    path = "resources/images/difficult/3-2.jpg"
    img = cv2.imread(path)
    plate, corners = locate_plate(img, visualize=True)