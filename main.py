import cv2

# Load image
img = cv2.imread("resources/images/easy/1-3.jpg")
# img = cv2.imread("resources/images/medium/2-1.jpg")
# img = cv2.imread("resources/images/difficult/3-2.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding to preprocess the image
# _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Invert the image if the text is black
# to ensure the text is white and the background is black
half = thresh.shape[0]*thresh.shape[1]*255/2
if thresh.sum() > half:
    thresh = cv2.bitwise_not(thresh)

# Erode and then dilate to make the characters more visible
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
eroded = cv2.erode(thresh, kernel, iterations=5)
dilated = cv2.dilate(eroded, kernel, iterations=15)


# Find contours
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# each element in borders is (x, y, w, h) of the bounding box of a character
borders = [cv2.boundingRect(contour) for contour in contours]
borders.sort(key=lambda x: x[0])

# # draw the borders on the image
# demo_tmp = cv2.cvtColor(dilated, cv2.COLOR_GRAY2RGB)
# for border in borders:
#     x, y, w, h = border
#     cv2.rectangle(demo_tmp, (x, y), (x+w, y+h), (0,0,255), 3)

# Extract the regions of interest
rois = [thresh[y:y+h, x:x+w] for x, y, w, h in borders if h > thresh.shape[0]*0.7]
# Pad the rois
rois = [cv2.resize(cv2.copyMakeBorder(roi, 10,10,10,10, cv2.BORDER_CONSTANT, value=(0,0,0)),
                    (350,700), interpolation=cv2.INTER_NEAREST) for roi in rois]
# rois = [cv2.copyMakeBorder(roi, 10,10,10,10, cv2.BORDER_CONSTANT, value=(0,0,0)) for roi in rois]
# Erode and then dilate to remove the noise
e_rois = [cv2.erode(roi, kernel, iterations=2) for roi in rois]
d_rois = [cv2.dilate(e_roi, kernel, iterations=4) for e_roi in e_rois]

# Visualize the segmented characters
for roi, e_roi, d_roi in zip(rois, e_rois, d_rois):
    # # compare the original roi and the processed roi
    # cv2.imshow('roi', roi)
    # cv2.imshow('e_roi', e_roi)
    cv2.imshow('d_roi', d_roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# cv2.namedWindow('img', 0)
# cv2.resizeWindow('img', int(600*img.shape[1]/img.shape[0]), 600)
# cv2.imshow('img', dilated)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
