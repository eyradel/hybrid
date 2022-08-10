# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 08:33:52 2022

@author: Delaeyram
"""

import cv2
cap = cv2.imread(r"C:\Users\Delaeyram\Downloads\dine.jpg")
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights","dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320,320),scale=1/255)
classes = []
with open("dnn_model/classes.txt","r") as file_object:
    for class_name in file_object.readlines():
        class_name  = class_name.strip()
        classes.append(class_name)
print(classes)
(class_ids,scores,bboxes) = model.detect(cap)
for class_id,score,bbox in zip(class_ids,scores,bboxes):
    (x,y,w,h) = bbox
    class_name = classes[class_id]
        
    cv2.putText(cap,class_name,(x,y-10),cv2.FONT_HERSHEY_PLAIN,2,(100,0,50),2)
    
    cv2.rectangle(cap,(x,y),(x+w,y+h),(100,0,50),2)
        
cv2.imshow("video",cap)
cv2.waitKey(0)
    
cv2.destroyAllWindows()