import cv2
import os
from locate import locate_plate
from rectify import rectify_plate
from segment import segment_chars
from recognize import recognize_char


for root, ds, fs in os.walk(R"resources\images"):
    for f in fs:
        path = os.path.join(root, f)
        img = cv2.imread(path)
        if f[0] == '1':
            chars = segment_chars(img, visualize=True, easy=True)
        else:
            plate, corners = locate_plate(img, visualize=False)
            plate = rectify_plate(plate, corners, visualize=False)
            chars = segment_chars(plate, visualize=True)

# path = "resources/images/medium/2-1.jpg"
# # path = "resources/images/difficult/3-1.jpg"
# img = cv2.imread(path)
# plate, corners = locate_plate(img, visualize=False)
# plate = rectify_plate(plate, corners, visualize=True)
# chars = segment_chars(plate, visualize=True)
