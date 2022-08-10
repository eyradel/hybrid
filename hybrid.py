# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 12:27:12 2022

@author: Delaeyram
"""

import streamlit as st
import cv2
st.markdown("Built by Eyram Dela")
run = st.checkbox("Run")
FRAME_WINDOW = st.image([])
video = cv2.VideoCapture(0)
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights","dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320,320),scale=1/255)
classes = []
with open("dnn_model/classes.txt","r") as file_object:
    for class_name in file_object.readlines():
        class_name  = class_name.strip()
        classes.append(class_name)
print(classes)

while run:
    sucess,frame =video.read()
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    (class_ids,scores,bboxes) = model.detect(frame)
    for class_id,score,bbox in zip(class_ids,scores,bboxes):
        (x,y,w,h) = bbox
        class_name = classes[class_id]
        
        cv2.putText(frame,class_name,(x,y-10),cv2.FONT_HERSHEY_PLAIN,2,(100,0,50),2)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(100,0,50),2)
    FRAME_WINDOW.image(frame)    
    
    


