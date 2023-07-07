import cv2, os
import numpy as np

if __name__ == '__main__':
    # 절대 경로 가져오기
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for i in range(1, 11):
        # 이미지 경로 탐색
        image_path = os.path.join(script_dir, 'images', 'box' + str(i) + '.jpg')

        img = cv2.imread(image_path, 0) # 0: gray scale로 이미지 불러오기
        img = cv2.resize(img, (256,256)) # 256x256 pixel로 resize
        
        # Gaussian Blur 적용 5x5 fields 마다 흐리게(Bluring) 만들어버림 -> 약한 경계선의 경우 문질러져서 사라지게 된다
        blur = cv2.GaussianBlur(img, ksize=(5,5), sigmaX=0)
        #  Canny Edge Detection/ 이미지, Threshold1: 작을 수록 선이 조금더 길게 나옴, Threshold2: 작을 수록 선이 더 많이 검출됨
        edged = cv2.Canny(blur, 10, 150)
        cv2.imshow('Edged', edged) # edge 찾은거 띄우기

        # morphology를 위한 kernel 제작 3x3의 kernel로 사각형(MORPH_RECT), 즉 커널이 전부 1로 채워짐
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        # 커널을 사용해 MORPH_CLOSE -> 커널에 맞게 주변 픽셀 다 선택해서 채우기 때문에 선이 두꺼워진다.
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    
        # 원본 이미지와 합쳐서 imshow()
        merged = np.hstack((img, closed))
        cv2.imshow('Probability hough line', merged)
        cv2.waitKey() # 아무 키 누르면 다음 반복문, 즉 다음 이미지 뜸
    cv2.destroyAllWindows()