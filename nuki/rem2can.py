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

    #이미지를 GRAY SCALE로 읽기
    input = cv2.imread(INPUT_PATH, 0)
    
    # 배경 제거, 이때 배경은 검정
    output = remove(input,
        bgcolor=[0,0,0, 255])
    
    # 배경 제거 결과 출력
    cv2.imwrite(OUTPUT_PATH, output)
    cv2.imshow('what', output)
    
    # Canny를 통해 외곽선만 검출(threshold는 통상적인 값, 추후 실험을 통해 변경 필요)
    # 이미지, Threshold1: 작을 수록 선이 조금더 길게 나옴, Threshold2: 작을 수록 선이 더 많이 검출됨
    nuki = cv2.Canny(output, 100, 200)

    # nuki 결과 출력
    
    cv2.imshow('nuki', nuki)

    # morphology를 위한 kernel 제작 nxn의 kernel로 사각형(MORPH_RECT), 즉 커널이 전부 1로 채워짐
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    # 커널을 사용해 MORPH_CLOSE -> 커널에 맞게 주변 픽셀 다 선택해서 채우기 때문에 선이 두꺼워진다.
    morhpology = cv2.morphologyEx(nuki, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite(CROP_PATH, morhpology)
    cv2.imshow('morphology', morhpology)

    cv2.waitKey()
