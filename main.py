import cv2

# Load image
img = cv2.imread("resources/images/easy/1-1.jpg")
# img = cv2.imread("resources/images/medium/2-1.jpg")
# img = cv2.imread("resources/images/difficult/3-2.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding to preprocess the image
# _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Apply dilation to make the characters more visible
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
dilated = cv2.dilate(thresh, kernel, iterations=1)



cv2.namedWindow('img', 0)
cv2.resizeWindow('img', int(600*img.shape[1]/img.shape[0]), 600)
cv2.imshow('img', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()