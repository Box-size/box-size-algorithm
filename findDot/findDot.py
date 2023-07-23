import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

#함수 정의 구간//////////////////////////
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
#이미지 좌표계 상의 가로, 세로, 높이를 추정하는 함수 
def calc_pixel_w_h(top, left, right, bottom):
    left_top = min(left, key=lambda p: p[1])
    left_bottom = max(left, key=lambda p: p[1])
    right_top = min(right, key=lambda p: p[1])
    right_bottom = max(right, key=lambda p: p[1])

    width = (math.sqrt((top[0] - left_top[0])**2 + (top[1] - left_top[1])**2) +
             math.sqrt((bottom[0] - right_bottom[0])**2 + (bottom[1] - right_bottom[1])**2)) / 2
    height = (math.sqrt((top[0] - right_top[0])**2 + (top[1] - right_top[1])**2) +
             math.sqrt((bottom[0] - left_bottom[0])**2 + (bottom[1] - left_bottom[1])**2)) / 2
    print(height/width)
    tall = (math.sqrt((left_top[0] - left_bottom[0])**2 + (left_top[1] - left_bottom[1])**2) + 
            math.sqrt((right_top[0] - right_bottom[0])**2 + (right_top[1] - right_bottom[1])**2)) / 2

    return width, height, tall

#//////////////////////////////////////
input_path = 'findDot/crops/crop.png'

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
       
top, bottom, left, right = classify_points(points)
plt.show()    

left_top = min(left, key=lambda p: p[1])
left_bottom = max(left, key=lambda p: p[1])
right_top = min(right, key=lambda p: p[1])
right_bottom = max(right, key=lambda p: p[1])

#이미지 꼭지점 좌표를 토대로 구한 가로, 세로, 높이
width, height, tall = calc_pixel_w_h(top, left, right, bottom)
#2D 이미지 좌표
image_points = np.array([[bottom[0], bottom[1]],
                         [left_bottom[0], left_bottom[1]],
                         [right_bottom[0], right_bottom[1]],
                         [left_top[0], left_top[1]], 
                         [right_top[0], right_top[1]],
                         [top[0], top[1]]], 
                         dtype=np.float32)
#3D 좌표계에 생성한 박스 좌표
object_points = np.array([[0, 0, 0],
                          [width, 0, 0],
                          [0, height, 0],
                          [width, 0, tall],
                          [0, height, tall],
                          [width, height, tall]],
                          dtype=np.float32)

#TODO: 카메라의 초점거리와 셀 크기를 알아오는 작업 필요
fx, fy, cx, cy = 944.4, 944.4, edges.shape[1] / 2, edges.shape[0] / 2

cameraMatrix = np.array([[fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]],
                        dtype=np.float32)

#외부 파라미터 추정
retval, rvec, tvec = cv2.solvePnP(object_points, image_points, cameraMatrix, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)

# 3D 좌표계 상에서 카메라의 위치와 방향 계산
rotation_matrix, _ = cv2.Rodrigues(rvec)
camera_position = -np.dot(rotation_matrix.T, tvec)

# 3D 그래프 생성
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 카메라의 위치와 방향 그리기
ax.quiver(camera_position[0], camera_position[1], camera_position[2], rvec[0], rvec[1], rvec[2])

#물체 위치 그리기
ax.scatter3D(object_points[:, 0], object_points[:, 1], object_points[:, 2])

# 그래프 표시
print(width, height, tall)
plt.show()


