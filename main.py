import cv2
import numpy as np
from util import get_limits, get_average_color, find_contours, get_nearest_color, add_to_sorted_contours

colors = {
    'B': [200, 50, 0],      # Blue
    'G': [0, 200, 0],        # Green
    'O': [0, 128, 240],      # Orange
    'R': [70, 0, 200],       # Red
    'W': [255, 255, 255],    # White
    'Y': [0, 255, 255]       # Yellow
}

cam = cv2.VideoCapture(0)
while True:
    color_contours = []
    ret, frame = cam.read()

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for color in colors.items():
        color_contours += find_contours(hsvImage, color[1])

    face_tiles = []

    sorted_contours = []
    for contour in color_contours:
        if cv2.contourArea(contour) < 2000 or cv2.contourArea(contour) > 10000:
            continue
        # face_tiles.append(nearest_color)
        x, y, w, h = cv2.boundingRect(contour)

        if w / h > 1.2 or h / w > 1.2:
            continue
        add_to_sorted_contours(sorted_contours, contour)

    i = 0
    for c in sorted_contours:
        i += 1
        x, y, w, h = cv2.boundingRect(c)
        average_color = get_average_color(frame, c)
        nearest_color = get_nearest_color(average_color)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (36, 255, 12), 2)
        cv2.putText(frame, "#{}".format(str(i) + " " + str(nearest_color)), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if len(face_tiles) == 9:
        print(face_tiles)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()

cv2.destroyAllWindows()
