from ultralytics import YOLO
from src.sort import *
import numpy as np
import cvzone
import math
import cv2

def load_model(url_yolo_model):
    model = YOLO(url_yolo_model)
    # cls = model.names
    cls = {
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck'
    }
    return cls, model

def run_detection(url_vid, url_yolo_model, mask_img_url, conf_level=0.3, win_name='recognition'):
    total_count = []

    classes, model = load_model(url_yolo_model)
    mask = cv2.imread(mask_img_url)
    mask = cv2.resize(mask, (854, 480))

    cam = cv2.VideoCapture(url_vid)
    # cam.set(cv2.CAP_PROP_POS_MSEC, 1*60*1000)

    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    while True:
        success, img = cam.read()
        # Drawing the line.
        cv2.line(img, (5, 280), (400, 280), (255, 255, 255), 2)

        image_region = cv2.bitwise_and(img, mask)
        results = model(image_region, stream=True)

        detections = np.empty((0, 5))

        for result in results:
            boxes = result.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0,0), 3) # Con cv2.

                conf = math.ceil((box.conf[0]*100))/100
                cls = int(box.cls[0])
                
                if cls in classes and conf >= conf_level:
                    # Bunding box.
                    cvzone.cornerRect(img, (x1, y1, x2-x1, y2-y1), t=2, colorC=(255, 0, 0), colorR=(255, 0, 0), l=1)
                    # Text rectangle.
                    cvzone.putTextRect(img, f'{classes[cls]} {conf}', (x1, y1-15), scale=1, thickness=1, colorR=(255, 0, 0))

                    detections = np.vstack((detections, np.array([x1, y1, x2, y2, conf])))

        result_tracker = tracker.update(detections)

        for resu in result_tracker:
            x1, y1, x2, y2, id = resu
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1

            center_x, center_y = x1 + w //2, y1+h//2
            
            # cvzone.cornerRect(img, (x1, y1, x2-x1, y2-y1), t=2, colorC=(255, 0, 0), colorR=(255, 0, 0), l=1)
            # cv2.circle(img, (center_x, center_y), 5, (255,0,0), cv2.FILLED)
            # cvzone.putTextRect(img, f'id {id}', (x1, y1-15), scale=1, thickness=1, colorR=(255, 0, 0))

            if 5 < center_x < 400 and 280 - 20 < center_y < 280 +20:
                # if the car touch the range of the line.
                cv2.line(img, (5, 280), (400, 280), (0, 255, 0), 2)
                if total_count.count(id) == 0:
                    total_count.append(id)

            cvzone.putTextRect(img, f'Counter of cars: {len(total_count)}', (50, 50), scale=1, thickness=1, colorR=(0, 0, 0))

        cv2.imshow(win_name, img)

        if cv2.waitKey(1) != -1:
            break

    # Destruye las ventanas y libera la camara.
    cam.release()
    cv2.destroyAllWindows()