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

    INPUT_NAME = 'box9.png'
    OUTPUT_NAME = 'box.png'
    CROP_NAME = 'crop.png'

    INPUT_PATH = os.path.join(INPUT_DIR, INPUT_NAME)
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    CROP_PATH = os.path.join(CROP_DIR, CROP_NAME)
    #NOTE: OUTPUT_PATH로 하면 디렉토리생성이 이미지 명으로 되고 얻어진 이미지가 저장이 안되어서 수정했습니다.
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if not os.path.exists(CROP_DIR):
            os.makedirs(CROP_DIR)
    # PATH 세팅 끝

    #원본 비율을 간직한 채 resize를 한 후, 남은 공간을 검은색으로 채우는 함수
    #https://engineer-mole.tistory.com/314
    def resize_ratio(pic):
        size=(256, 256)
        base_pic=np.zeros((size[1],size[0],3),np.uint8)
        pic1=cv2.imread(pic,cv2.IMREAD_COLOR)
        h,w=pic1.shape[:2]
        ash=size[1]/h
        asw=size[0]/w
        if asw<ash:
            sizeas=(int(w*asw),int(h*asw))
        else:
            sizeas=(int(w*ash),int(h*ash))
        pic1 = cv2.resize(pic1,dsize=sizeas)
        base_pic[int(size[1]/2-sizeas[1]/2):int(size[1]/2+sizeas[1]/2),
        int(size[0]/2-sizeas[0]/2):int(size[0]/2+sizeas[0]/2),:]=pic1
        return base_pic

    # 이미지를 GRAY SCALE로 읽기
    input = cv2.imread(INPUT_PATH, 0)
    # NOTE: 원본 비율 보존한 채 resize 해결했습니다.
    input = resize_ratio(INPUT_PATH)
    # resize_ratio 에서 IMREAD_COLOR로 불러왔기 때문에, 그레이 스케일 변경
    input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)

    # 명암비 alpha 0이면 그대로, 양수일수록 명암비가 커진다.
    alpha = 0.5
    input = np.clip((1+alpha) * input - 128 * alpha, 0, 255).astype(np.uint8)
    
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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # 커널을 사용해 MORPH_CLOSE -> 커널에 맞게 주변 픽셀 다 선택해서 채우기 때문에 선이 두꺼워진다.
    morhpology = cv2.morphologyEx(nuki, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite(CROP_PATH, morhpology)
    cv2.imshow('morphology', morhpology)

    cv2.waitKey()
