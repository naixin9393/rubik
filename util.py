import cv2
import numpy as np


colors = {
    'B': [200, 50, 0],      # Blue
    'G': [0, 200, 0],        # Green
    'O': [0, 128, 240],      # Orange
    'R': [70, 0, 200],       # Red
    'W': [255, 255, 255],    # White
    'Y': [0, 255, 255]       # Yellow
}


def get_limits(color):
    # detecting white color
    if color == [255, 255, 255]:
        lower_white = np.array([0, 0, 120], dtype=np.uint8)
        upper_white = np.array([255, 100, 255], dtype=np.uint8)
        return lower_white, upper_white

    colorArray = np.uint8([[color]])
    hsvColor = cv2.cvtColor(colorArray, cv2.COLOR_BGR2HSV)

    lowerLimit = hsvColor[0][0][0] - 10, 100, 100
    upperLimit = hsvColor[0][0][0] + 10, 255, 255

    lowerLimit = np.array(lowerLimit, dtype=np.uint8)
    upperLimit = np.array(upperLimit, dtype=np.uint8)

    return lowerLimit, upperLimit


def get_average_color(image, contour):
    # Create a mask for the contour
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [contour], (255))

    # Use the mask to extract the region of interest
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Calculate the average color inside the masked area
    mean_color = cv2.mean(masked_image, mask=mask)[:3]

    return tuple(map(int, mean_color))  # Return as integer tuple


def find_contours(hsvImage, color):

    lowerLimit, upperLimit = get_limits(color=color)

    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = contours[0] if len(contours) == 2 else contours[1]

    return contours


def get_nearest_color(color):
    min_distance = 999999
    nearest_color = None
    for key, value in colors.items():
        distance = np.linalg.norm(np.array(color) - np.array(value))
        if distance < min_distance:
            min_distance = distance
            nearest_color = key
    return nearest_color


def add_to_sorted_contours(sorted_contours, contour):
    sorted_contours.append(contour)
