import cv2
import numpy as np
from typing import Optional
from utils import visualize_resize


def isQuadrangle(contour: np.ndarray) -> Optional[list[np.ndarray]]:
    """Checks if the contour is a quadrangle. Returns the four vertices if yes and None otherwise.

    Args:
        contour (np.ndarray): The contour to check.

    Returns:
        Optional[list[np.ndarray]]: The four vertices if the contour is a quadrangle and None otherwise.
    """

    minRec = cv2.minAreaRect(contour)
    vertices = np.int0(cv2.boxPoints(minRec))
    contour_area = cv2.contourArea(contour)

    # Greedily adjust the vertices to make the quadrangle approximate the contour
    prev_area = cv2.contourArea(vertices)
    while True:
        for i, v in enumerate(vertices):
            cur_min = abs(cv2.contourArea(vertices) - contour_area)
            # print("cur_min", cur_min)
            # Loop through the neighbors as candidates in search of a better vertex
            for cand_v in [
              np.array([v[0]+10,v[1]]), np.array([v[0]-10,v[1]]),
              np.array([v[0],v[1]+10]), np.array([v[0],v[1]-10])]:
                # Compute the area of the quadrangle with the candidate vertex
                cand_area = cv2.contourArea(
                    np.concatenate((vertices[:i], [cand_v], vertices[i+1:])))
                this_delta = abs(cand_area - contour_area)
                if this_delta < cur_min:
                    # print("new", this_delta, cur_min)
                    cur_min = this_delta
                    # print(vertices[i], cand_v)
                    vertices[i] = cand_v
            
        cur_area = cv2.contourArea(vertices)

        print(vertices)
        print(abs(cur_area - contour_area))
        if abs(cur_area - prev_area) < 10:  # converged
            break
        prev_area = cur_area
    print(vertices)
    return vertices


path = "resources/images/medium/2-1.jpg"
path = "resources/images/difficult/3-3.jpg"
img = cv2.imread(path)
# visualize_resize(img, "img", height=500, close=False)

# Apply median blur to smoothen the image
blur = cv2.medianBlur(img, 51)
# visualize_resize(blur, "blur", height=500)

# Convert to HSV to filter out the blue and green regions
hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
blue_mask = cv2.inRange(hsv, np.array([100, 110, 110]), np.array([110, 255, 255]))
green_mask = cv2.inRange(hsv, np.array([35, 20, 46]), np.array([77, 255, 255]))
plate_mask = cv2.bitwise_or(blue_mask, green_mask)
# visualize_resize(plate_mask, "plate_mask", height=500, close=False)

# # Erode to remove noise
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
# eroded = cv2.erode(plate_mask, kernel, iterations=3)
# visualize_resize(eroded, "eroded", height=500, close=False)

# Dilate to make each object more connected
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
dilated = cv2.dilate(plate_mask, kernel, iterations=8)
# visualize_resize(dilated, "dilated", height=500)

# Find contours
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img, contours, -1, (0,0,255), 5)
# visualize_resize(img, "contours", height=500)

# # Filter for quadrangles
# recs = []
# for contour in contours:
#     rec = cv2.minAreaRect(contour)
#     cv2.drawContours(img, [np.int0(cv2.boxPoints(rec))], 0, (0,0,255), 5)
#     vertices = isQuadrangle(contour)
#     if vertices is not None:
#         recs.append(vertices)
#         # cv2.drawContours(img, [np.array([vertices[3], vertices[0]])], 0, (0,255,0), 5)
#         cv2.drawContours(img, [vertices], 0, (0,255,0), 5)
# visualize_resize(img, "borders", height=500)

# canvas = np.zeros_like(img)
# cv2.drawContours(canvas, contours, -1, (255,255,255), 5)
# visualize_resize(canvas, "contours", height=500)

# Filter for quadrangles
corners = []
for contour in contours:
    polygon = cv2.approxPolyDP(contour, 0.05*cv2.arcLength(contour, True), True)
    if len(polygon) == 4:
        corners.append(polygon)
        # cv2.drawContours(img, [polygon], 0, (0,255,0), 5)

demo_img = img.copy()
cv2.drawContours(demo_img, corners, -1, (0,0,255), 5)
visualize_resize(demo_img, "corners", height=500)

corners = max(corners, key=cv2.contourArea)
minx = min(corners[:,0,0])
miny = min(corners[:,0,1])
maxx = max(corners[:,0,0])
maxy = max(corners[:,0,1])
roi = img[miny:maxy, minx:maxx]
visualize_resize(roi, "roi", height=500)


# # Filter for rectangles
# recs = []
# for contour in contours:
#     rec = cv2.minAreaRect(contour)
#     # rec = cv2.boundingRect(contour)
#     # x, y, w, h = rec
#     box = cv2.boxPoints(rec)
#     contour_area = cv2.contourArea(contour)
#     rec_area = cv2.contourArea(box)
#     # rec_area = w * h
#     # print(w, h)
#     if contour_area / rec_area >= 0.9:
#         cv2.drawContours(img, [contour], 0, (0,255,0), 5)
#         # cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 5)
#         cv2.drawContours(img, [np.int0(box)], 0, (0,0,255), 5)
#         recs.append([rec])
#         print(rec)
# visualize_resize(img, "borders", height=500)



# # # # Use Harris corner detector to detect the corners
# # # corners = cv2.cornerHarris(dilated, 2, 3, 0.04)
# # # corners = cv2.dilate(corners, None)
# # # img[corners > 0.1*corners.max()] = [0,0,255]
# # # visualize_resize(img, "corners", height=500)

# # # detect edges using Canny
# # edges = cv2.Canny(dilated, 15, 50, apertureSize=3)
# # visualize_resize(edges, "edges", height=500)

# # # apply Hough transform
# # lines = cv2.HoughLines(canvas, rho=1, theta=np.pi / 180, threshold=500)
# # print(len(lines))
# # if isinstance(lines, np.ndarray):
# #     # draw detected lines
# #     for line in lines:
# #         # compute endpoints
# #         rho, theta = line[0]
# #         a = np.cos(theta)
# #         b = np.sin(theta)
# #         x0 = a * rho
# #         y0 = b * rho
# #         x1 = int(x0 + 10000 * (-b))
# #         y1 = int(y0 + 10000 * a)
# #         x2 = int(x0 - 10000 * (-b))
# #         y2 = int(y0 - 10000 * a)
# #         cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 5)

# # visualize_resize(img, "img1", height=500)