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
#YOLO폴더
- YOLO v5를 이용한 박스 이미지만 추출하는 코드
- 모델 안정성 개선 필요
- detect.py 코드 실행을 통해 구동