import cv2
from segment import segment_chars
from recognize import recognize_char


path = "resources/images/easy/1-2.jpg"
img = cv2.imread(path)
chars = segment_chars(img, visualize=True)
