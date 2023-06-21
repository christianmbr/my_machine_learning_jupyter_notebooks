from src.car import *

url_vid = '../images/my_videos/car_vid480.mp4'
yolo_model = '../yolo_model/yolov8n.pt'
mask_url = './img/mask.jpg'

run_detection(url_vid, yolo_model, mask_url)