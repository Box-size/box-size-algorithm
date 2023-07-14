import cv2
import numpy as np
import matplotlib.pyplot as plt

input_path = 'findDot/crops/crop4.png'

edges = cv2.imread(input_path)
edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
hexagon_contours = []

# 육각형 윤곽선 필터링
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(approx) == 6:
        hexagon_contours.append(approx)

plt.imshow(edges)

# 근사화된 윤곽선에서 각 꼭지점의 좌표 추출 및 표시
for hexagon in hexagon_contours:
    for point in hexagon:
        x, y = point[0]
        plt.scatter(x, y, color='red', s=10)

plt.show()