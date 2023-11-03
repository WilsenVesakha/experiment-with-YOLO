from ultralytics import YOLO
import cv2
import os
import json
model = YOLO('yolov8n.pt')

def calculateIOU(box1, box2):
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2
    xA = max(x1, a1)
    yA = max(y1, b1)
    xB = min(x2, a2)
    yB = min(y2, b2)
    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (a2 - a1 + 1) * (b2 - b1 + 1)
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def getAnnotationCoordinatesData(data,img):
    res=[]
    for c in range(len(data)):
        if data[c]['image_id']==int(img.split('.')[0]):
            res.append(data[c]['bbox'][0])
            res.append(data[c]['bbox'][1])
            res.append(data[c]['bbox'][0]  + data[c]['bbox'][2] )
            res.append(data[c]['bbox'][1] + data[c]['bbox'][3] )
    return res


f = open(PATH_TO_JSON_LABELS)
data = json.load(f)
valData = []
for i in data['annotations']:
    valData.append(i)
    
dataset = PATH_TO_DATASET
imgList = os.listdir(dataset)
truePositives=0
totalDetections=0
iouThreshold=0.5
for x in imgList:
    results = model(dataset+x, save=True)
    if len(results)==1 and len(results[0].boxes.xyxy)==1: #ensure that an object is detected and the detection result is only 1
        cords = results[0].boxes.xyxy[0].tolist()
        class_id = results[0].boxes.cls[0].item()
        conf = results[0].boxes.conf[0].item()
        res= getAnnotationCoordinatesData(valData,x)
        if conf > 0.5:
            IoU = calculateIOU( cords, res)
            truePositives = truePositives+ 1 if IoU >= iouThreshold else truePositives
            totalDetections += 1
        #resAnnotation=cv2.rectangle(cv2.imread(dataset+x), (int(res[0]),int(res[1])), (int(res[2]),int(res[3])), (255,0,0), 2)
        #resPred=cv2.rectangle(resAnnotation, (int(cords[0]),int(cords[1])), (int(cords[2]),int(cords[3])), (0,255,0), 4)
        #cv2.imshow(x,resPred)
        #cv2.waitKey()










