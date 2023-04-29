import cv2
import numpy as np
from utils import visualize_resize


path = "resources/images/medium/2-3.jpg"
# path = "resources/images/difficult/3-1.jpg"
img = cv2.imread(path)
# visualize_resize(img, "img", height=500, close=False)

# Convert to HSV to filter out the blue and green regions
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
blue_mask = cv2.inRange(hsv, np.array([100, 110, 110]), np.array([110, 255, 255]))
green_mask = cv2.inRange(hsv, np.array([35, 20, 46]), np.array([77, 255, 255]))
plate_mask = cv2.bitwise_or(blue_mask, green_mask)
visualize_resize(plate_mask, "plate_mask", height=500, close=False)

# Apply median blur to remove noise
blur = cv2.medianBlur(plate_mask, 101)
visualize_resize(blur, "blur", height=500, close=False)

# Dilate to make each object more connected
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
dilated = cv2.dilate(blur, kernel, iterations=20)
visualize_resize(dilated, "dilated", height=500)

# Find contours
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img, contours, -1, (0,0,255), 5)
# visualize_resize(img, "contours", height=500)

for contour in contours:
    polygon = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
    cv2.drawContours(img, [polygon], 0, (0,0,255), 5)
    print(len(polygon))
visualize_resize(img, "contours", height=500)


# # Filter for rectangles
# recs = []
# for contour in contours:
#     # rec = cv2.minAreaRect(contour)
#     rec = cv2.boundingRect(contour)
#     x, y, w, h = rec
#     # box = cv2.boxPoints(rec)
#     area = cv2.contourArea(contour)
#     rec_area = w * h
#     print(w, h)
#     # if w / h > 2 and area / rec_area >= 0.9:
#     cv2.drawContours(img, [contour], 0, (0,0,255), 5)
#     cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 5)
#     # cv2.drawContours(img, [np.int0(box)], 0, (0,0,255), 5)
#     recs.append([rec])
#     print(rec)
# visualize_resize(img, "borders", height=500)



# # # Use Harris corner detector to detect the corners
# # corners = cv2.cornerHarris(dilated, 2, 3, 0.04)
# # corners = cv2.dilate(corners, None)
# # img[corners > 0.1*corners.max()] = [0,0,255]
# # visualize_resize(img, "corners", height=500)

# # detect edges using Canny
# edges = cv2.Canny(dilated, 15, 50, apertureSize=3)
# visualize_resize(edges, "edges", height=500)

# # apply Hough transform
# lines = cv2.HoughLines(canvas, rho=1, theta=np.pi / 180, threshold=500)
# print(len(lines))
# if isinstance(lines, np.ndarray):
#     # draw detected lines
#     for line in lines:
#         # compute endpoints
#         rho, theta = line[0]
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a * rho
#         y0 = b * rho
#         x1 = int(x0 + 10000 * (-b))
#         y1 = int(y0 + 10000 * a)
#         x2 = int(x0 - 10000 * (-b))
#         y2 = int(y0 - 10000 * a)
#         cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 5)

# visualize_resize(img, "img1", height=500)