# box-size-algorithm

사진에서 박스를 탐색하고, 실제 박스의 가로, 세로, 높이를 계산합니다.

> CJ대한통운 미래기술 챌린지 2023(Box.size 팀)

## Installation

### Activate Virtual Environment

```sh
$ activate_venv.sh
```

If you want to deactivate virtual environment, run `deactivate_venv.sh`.

> If Windows, please use `.bat` files.

### Install Dependency

```sh
$ pip install -r requirements.txt
```
<details>
<summary>segment-anything은 사용하지 않기로 했습니다.</summary>

* ~~[CUDA Toolkit 11.8.0](https://developer.nvidia.com/cuda-11-8-0-download-archive)~~
* ~~[cuDNN v8.8.0 for CUDA 11.x](https://developer.nvidia.com/rdp/cudnn-archive)~~

> 


### Import Model

~~[Download sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth), and place to `/models`.~~

> `sam_vit_h_4b8939.pth` is a model provided by Facebook Research.

</details>

## YOLO v5

### Run

```sh
$ python yolo/detect.py
```

### Detail

YOLO v5를 이용한 박스 이미지만 추출하는 코드

추출된 이미지는 /result/exp/crops 에 있습니다.

## nuki

### Run

```sh
$ python nuki/rem2can.py
```

### Detail

YOLO v5를 이용해 추출된 이미지를 
rembg라이브러리를 이용해 배경을 제거하고 
Canny메소드를 이용해 외곽선만 검출한 코드

추출된 이미지는 nuki/crops 에 있습니다.

## YOLO v8

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

## findDot

### Run

```sh
$ python findDot/findDot.py
```

### Detail

nuki에서 얻어진 crops 폴더의 이미지를 바탕으로 박스 꼭지점 좌표를 알아내는 코드

hexagon_contours에는 box 윤곽선의 배열이 저장(numpy)

각 윤곽선의 좌표는 (x, y) 형태로 표현, 이미지 상에서 박스의 각 꼭지점 위치를 나타냄