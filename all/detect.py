from ultralytics import YOLO
import cv2

def detect(image_path):
    model = YOLO('yolo-v8/detect_model.pt')
    source = image_path  #'all/images/box8.jpg'
    model.predict(source, imgsz=640, conf=0.5, max_det=1)
    results = model(source)
    boxes = results[0].boxes
    box = boxes[0]  # returns one box
    
    res = results[0].plot(boxes=False)
    lt = box.xyxy[0][:2].tolist() # lefttop
    rb = box.xyxy[0][2:].tolist() # rightbottom

    res_crop = res[int(lt[1]):int(rb[1]), int(lt[0]):int(rb[0])]

    return res_crop, box.xyxy
    #cv2.imshow("detect, crop",res_crop)
    #cv2.waitKey()