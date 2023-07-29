# box-size-algorithm

사진에서 박스를 탐색하고, 실제 박스의 가로, 세로, 높이를 계산합니다.

> CJ대한통운 미래기술 챌린지 2023(Box.size 팀)

## Installation

### Activate Virtual Environment

```sh
$ source ./activate_venv.sh
```

If you want to deactivate virtual environment, run `deactivate`.

> If Windows, please use `.bat` files.

### Install Dependency

```sh
$ pip install -r requirements.txt
```

## 이미지에서 박스 부분 잘라내기(YOLO v8)

### Install Module

```sh
$ pip install -U ultralytics
```

### Run

```sh
$ yolo predict model=detect_model.pt imgsz=640 conf=0.5 source=images save_crop=True
```
conf 조절을 통해 특정 confidence 값 이상만 추출하도록 할 수 있습니다.

### detail

검출 결과는 yolo-v8/runs/detect/predict에 있고,

박스만 추출된 파일은 yolo-v8/runs/detect/predict/crops/0 에 있습니다.

YOLO v8 커스텀 모델 학습.ipynb파일로 jupyter notebook을 통해 

직접 YOLO v8 모델 학습도 가능 합니다.

## 박스 이미지의 배경 제거 및 Edge Detection

### Run

```sh
$ python nuki/rem2can.py
```

### Detail

YOLO v5를 이용해 추출된 이미지를 
rembg라이브러리를 이용해 배경을 제거하고 
Canny메소드를 이용해 외곽선만 검출한 코드

추출된 이미지는 nuki/crops 에 있습니다.

## 최종 계산

### Run

```sh
$ python findDot/findDot.py
```

### Detail

nuki에서 얻어진 crops 폴더의 이미지를 바탕으로 박스 꼭지점 좌표를 알아내어 실제 상자의 크기를 구하는 코드

hexagon_contours에는 box 윤곽선의 배열이 저장(numpy)

각 윤곽선의 좌표는 (x, y) 형태로 표현, 이미지 상에서 박스의 각 꼭지점 위치를 나타냄

이후 초점거리, 원본사진의 중점, 2D좌표, 이미자 상의 박스 크기를 이용해 임의로 정한 3D좌표를 이용해 카메라 외부 파라미터(rvec, tvec)을 구함

Rodrigues(rvec)으로 진짜 카메라 정보를 얻은 후, 실제 월드 좌표계의 박스 한 점과, 카메라의 좌표를 구한후 둘 사이 거리 구함

카메라 초점거리 : 실제 카메라와 물체간 거리 = 이미지 상 박스 크기 : 실제 박스크기 비례식을 이용해 박스 크기 산출

모든 계산 식과 내용은 https://darkpgmr.tistory.com/153 참고