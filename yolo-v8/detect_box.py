from ultralytics import YOLO

model = YOLO('yolo-v8/detect_model.pt')
source = 'yolo-v8/images/box9.jpg'
model.predict(source, save=True, imgsz=640, conf=0.5, save_crop=True, max_det=1)
results = model(source)
boxes = results[0].boxes
box = boxes[0]  # returns one box
print(box.xyxy)
#549 463
#188.8067, 27.9195, 9번기준

#TODO: 원본 이미지로부터 crop이미지가 얼마나 떨어졌는가를 나타내는 box.xyxy를 findDot.py로 변수를 보내줘야 함
