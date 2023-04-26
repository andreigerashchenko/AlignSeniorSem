"""File containing functions for identifying the horizon."""

import cv2
import numpy as np
from scipy.interpolate import splprep, splev

def find_horizon(img: cv2.Mat, min_avg_height: float = 0.3, max_avg_height: float = 0.9, roughness_threshold: float = 0.1) -> np.ndarray:
    """Returns the horizon of an image as a spline."""
    # Resize the image if it's larger than 1280x720
    if img.shape[0] > 720 or img.shape[1] > 1280:
        scale_factor = min(1280 / img.shape[1], 720 / img.shape[0])
        img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur the image to reduce noise
    gray = cv2.GaussianBlur(gray, (9, 15), 0)

    # Apply thresholding
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Apply canny edge detection
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

    # Find contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Find the contour with the longest length that can be represented by a spline
    max_length = 0
    horizon_contour = None
    x_crossings = 0

    for cnt in contours:
        # Check if the contour has enough points to be represented by a spline
        if len(cnt) >= 50:
            # Get the average height of the contour
            avg_height = np.mean(cnt[:, 0, 1])
            # Skip the contour if it's not in the right height range
            if avg_height < min_avg_height * img.shape[0] or avg_height > max_avg_height * img.shape[0]:
                continue

            # TODO: Check if the contour is too rough
            
            # Fit a spline to the contour
            x, y = cnt.squeeze().T
            tck, u = splprep([x, y], s=0, k=min(3, len(x)-1))

            # Compute the length of the spline
            length = 0
            for i in range(len(x)-1):
                p1 = splev(u[i], tck)
                p2 = splev(u[i+1], tck)
                length += np.linalg.norm(np.array(p2)-np.array(p1))

            # Check if the spline is the longest so far and doesn't cross over the same X coordinate more often than the previous best
            if length > max_length:
                # Count the number of times the spline crosses the same X coordinate
                crossings = 0
                for i in range(len(x)-1):
                    p1 = splev(u[i], tck)
                    p2 = splev(u[i+1], tck)
                    if p1[0] < 0.5 * img.shape[1] and p2[0] > 0.5 * img.shape[1] or p1[0] > 0.5 * img.shape[1] and p2[0] < 0.5 * img.shape[1]:
                        crossings += 1

                # Only save the spline if it crosses the same X coordinate less often than the previous best
                if crossings <= x_crossings:
                    x_crossings = crossings
                    max_length = length
                    horizon_contour = cnt
    
    if horizon_contour is not None:
        return horizon_contour
    return None

def horizon_critical_points(img: cv2.Mat, horizon: np.ndarray):
    """Fits a spline to the horizon and returns the critical points of the spline."""
    # Fit a spline to the horizon
    x, y = horizon.squeeze().T
    tck, u = splprep([x, y], s=0, k=5)

    # Compute the critical points of the spline
    critical_points = []
    for i in range(len(x)-1):
        p1 = splev(u[i], tck)
        p2 = splev(u[i+1], tck)
        if p1[0] < 0.5 * img.shape[1] and p2[0] > 0.5 * img.shape[1] or p1[0] > 0.5 * img.shape[1] and p2[0] < 0.5 * img.shape[1]:
            critical_points.append(p1)
    return critical_points
