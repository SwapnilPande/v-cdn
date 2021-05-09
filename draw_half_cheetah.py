import cv2
import numpy as np


img = cv2.imread("cheetah.png")

# Center coordinates
joint_coords = [
    (168, 330),
    (147, 370),
    (127, 412),
    (334, 342),
    (334, 385),
    (355, 420),
    (250, 305),
]

edge_set = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]





# Radius of circle
radius = 10

# Blue color in BGR
color = (0, 0, 255)

# Line thickness of 2 px
thickness = 3

# Using cv2.circle() method
# Draw a circle with blue line borders of thickness of 2 px

for coord in joint_coords:
    image = cv2.circle(img, coord, radius, color, thickness)

thickness = 3

for i in range(7):
    for j in range(7):
        if(edge_set[7*i + j] == 1):
            start_point = joint_coords[i]
            end_point = joint_coords[j]

            vec = (np.array(end_point) - np.array(start_point))
            vec = vec/np.linalg.norm(vec)

            start_point = start_point + vec*(radius + thickness)
            start_point = tuple(start_point.astype(np.int))

            end_point = end_point - vec*(radius + thickness)
            end_point = tuple(end_point.astype(np.int))


            image = cv2.arrowedLine(image, start_point, end_point,
                                        color, thickness)

cv2.imwrite("new.png", img)