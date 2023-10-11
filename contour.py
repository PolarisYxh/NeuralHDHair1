import cv2

# Load the image
img = cv2.imread('path/to/image')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold the image
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find the contours in the image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate over each contour and approximate it
for contour in contours:
    # Calculate epsilon based on the contour perimeter
    epsilon = 0.01 * cv2.arcLength(contour, True)

    # Approximate the contour
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Draw the approximated contour on the original image
    cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)

# Display the image
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()