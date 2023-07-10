import numpy as np
import cv2
from rembg import remove

input_path = 'nuki/images/box1.png'
output_path = 'nuki/output/box.png'
crop_path = 'nuki/crops/crop.png'

#이미지를 읽어와서 배경 제거, 이때 배경은 흰색으로 씌우기
input = cv2.imread(input_path)
output = remove(input,
    alpha_matting=True,
    bgcolor=[255,255,255,255])

cv2.imwrite(output_path, output)
#배경이 제거된 이미지만 결과 출력
cv2.imshow('what', output)

#배경이 제거된 이미지 grayscale
img = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
cv2.imshow('src', img)
cv2.waitKey()

#Canny를 통해 외곽선만 검출(threshold는 통상적인 값, 추후 실험을 통해 변경 필요)
nuki = cv2.Canny(img, 50, 150)

cv2.imwrite(crop_path, nuki)
