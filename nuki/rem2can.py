import numpy as np
import cv2, os
from rembg import remove

if __name__ == '__main__':
    # 절대 경로 가져오기
    FILE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # PATH 세팅
    INPUT_DIR = os.path.join(FILE_DIR, 'images')
    OUTPUT_DIR =  os.path.join(FILE_DIR, 'outputs')
    CROP_DIR = os.path.join(FILE_DIR, 'crops')

    INPUT_NAME = 'box1.png'
    OUTPUT_NAME = 'box.png'
    CROP_NAME = 'crop.png'

    INPUT_PATH = os.path.join(INPUT_DIR, INPUT_NAME)
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    CROP_PATH = os.path.join(CROP_DIR, CROP_NAME)

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    if not os.path.exists(CROP_PATH):
            os.makedirs(CROP_PATH)
    # PATH 세팅 끝

    #이미지를 읽어와서 배경 제거, 이때 배경은 흰색으로 씌우기
    input = cv2.imread(INPUT_PATH)
    
    output = remove(input,
        alpha_matting=True,
        bgcolor=[0,0,0,0])

    cv2.imwrite(OUTPUT_PATH, output)
    #배경이 제거된 이미지만 결과 출력
    cv2.imshow('what', output)

    #배경이 제거된 이미지 grayscale
    img = cv2.imread(OUTPUT_PATH, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('src', img)
    
    #Canny를 통해 외곽선만 검출(threshold는 통상적인 값, 추후 실험을 통해 변경 필요)
    nuki = cv2.Canny(img, 10, 150)

    cv2.imwrite(CROP_PATH, nuki)
    cv2.imshow('nuki', nuki)

    cv2.waitKey()
