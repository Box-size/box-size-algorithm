from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
from rembg import remove

input_path = 'nuki/images/box9.png'
output_path = 'nuki/output/box.png'
crop_path = 'nuki/crops/crop.png'

input = cv2.imread(input_path)
output = remove(input,
    alpha_matting=True,
    bgcolor=[255,255,255,255])

cv2.imwrite(output_path, output)
cv2.imshow('what', output)

img = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
cv2.imshow('src', img)
cv2.waitKey()
nuki = cv2.Canny(img, 50, 150)

cv2.imwrite(crop_path, nuki)
