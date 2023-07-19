import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def classify_points(points):
    # y 좌표가 가장 낮은 점을 찾아 맨 위 점으로 설정
    top_point = min(points, key=lambda p: p[1])
    points.remove(top_point)

    # y 좌표가 가장 높은 점을 찾아 맨 밑 점으로 설정
    bottom_point = max(points, key=lambda p: p[1])
    points.remove(bottom_point)

    # x 좌표가 가장 작은 두 점을 찾아 왼쪽 점으로 설정
    left_points = sorted(points, key=lambda p: p[0])[:2]
    for p in left_points:
        points.remove(p)

    # 남은 두 점은 오른쪽 점으로 설정
    right_points = points

    return top_point, bottom_point, left_points, right_points

input_path = 'findDot/crops/crop9.png'

edges = cv2.imread(input_path)
edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
hexagon_contours = []

# 육각형 윤곽선 필터링
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(approx) <= 6:
        hexagon_contours.append(approx)

plt.imshow(edges)

# 근사화된 윤곽선에서 각 꼭지점의 좌표 추출 및 표시
points = []
for hexagon in hexagon_contours:
    for point in hexagon:
        x, y = point[0]
        plt.scatter(x, y, color='red', s=10)
        points.append((x, y))

print(points)        
top, bottom, left, right = classify_points(points)
plt.show()