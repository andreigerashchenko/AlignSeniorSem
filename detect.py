import cv2
import numpy as np

# Load the image
img = cv2.imread('examples\R0011132.JPG')

# Resize the image if it's larger than 1280x720
if img.shape[0] > 720 or img.shape[1] > 1280:
    scale_factor = min(1280 / img.shape[1], 720 / img.shape[0])
    img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply canny edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Find contours
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the contour with the longest length that approximates a sine wave
max_length = 0
horizon_contour = None
for cnt in contours:
    # Approximate the contour with a polygon
    epsilon = 0.1 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    # Check if the polygon has enough points to approximate a sine wave
    if len(approx) >= 20:
        # Fit a sine wave to the polygon
        x, y = approx.squeeze().T
        params = cv2.fitLine(approx, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = params
        a = vy / vx
        b = y0 - a * x0
        sine_y = a * x + b

        # Compute the mean squared error between the polygon and the sine wave
        mse = np.mean((y - sine_y) ** 2)

        # Check if the mean squared error is below a threshold
        if mse < 100:
            # Compute the length of the contour
            length = cv2.arcLength(cnt, closed=True)

            # Check if the contour is the longest so far
            if length > max_length:
                max_length = length
                horizon_contour = cnt

# Draw the contour on the image
cv2.drawContours(img, [horizon_contour], -1, (0, 255, 0), 2)

# Display the result
cv2.imshow('Horizon contour', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
