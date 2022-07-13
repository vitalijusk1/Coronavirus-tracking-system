import numpy as np
import cv2
import os # type: ignore
import imutils
import sys
import tensorflow 
from tensorflow.keras.models import load_model 
import tkinter as tk
from tkinter import *
import threading
import time




NMS_THRESHOLD=0.8
MIN_CONFIDENCE=0.

def pedestrian_detection(image, model, layer_name, personidz=0):
    (H, W) = image.shape[:2]
    results = []


    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    model.setInput(blob)
    # print(layer_name)
    layerOutputs = model.forward(layer_name)

    boxes = []
    centroids = []
    confidences = []

    for output in layerOutputs:
        for detection in output:

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if classID == personidz and confidence > MIN_CONFIDENCE:

                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))
    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idzs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)
    # ensure at least one detection exists
    if len(idzs) > 0:
        # loop over the indexes we are keeping
        for i in idzs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # update our results list to consist of the person
            # prediction probability, bounding box coordinates,
            # and the centroid
            res = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(res)
    # return the list of results
    
    return results



labelsPath = "C:/Users/kompiuteris/Desktop/python/room controlas/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

weights_path = "C:/Users/kompiuteris/Desktop/python/room controlas/yolov4-tiny.weights"
config_path = "C:/Users/kompiuteris/Desktop/python/room controlas/yolov4-tiny.cfg"

model = cv2.dnn.readNetFromDarknet(config_path, weights_path)

# model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
# print(model)

persons = 0

def aptikimas():
    layer_name = model.getLayerNames()
    layer_name = [layer_name[i - 1] for i in model.getUnconnectedOutLayers()]
    cap = cv2.VideoCapture("C:/Users/kompiuteris/Desktop/python/room controlas/vidcut.mp4")
    # cap = cv2.VideoCapture(0)
    writer = None
    counter = 0
    while True:
        counter = 30
        (grabbed, image) = cap.read()

        # print('a')
        if counter%30==0:
            # print(grabbed)
            if not grabbed:
                cap.release()
                cv2.destroyAllWindows()
                break
            image = imutils.resize(image, width=700)
            results = pedestrian_detection(image, model, layer_name,
                personidz=LABELS.index("person"))

            for res in results:
                cv2.rectangle(image, (res[1][0],res[1][1]), (res[1][2],res[1][3]), (0, 255, 0), 2)
                cv2.putText(image, f'Total Persons : {len(results)}', (40,70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
            # print(len(results))
            persons = len(results)
            if queue_aptikimas.empty():
                queue_aptikimas.put(persons)
            
        cv2.imshow("Detection",image)
        key = cv2.waitKey(1)
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            break
        
        counter += 1


    
has_mask = 0




def facemask():
    size = 4
    input_shape = 224
    labels_dict={0:'without mask',1:'mask'}
    color_dict={0:(0,0,255),1:(0,255,0)}
    webcam = cv2.VideoCapture("C:/Users/kompiuteris/Desktop/python/room controlas/kaukecut.mp4") #Use video file
    #webcam = cv2.VideoCapture(0)
    classifier = cv2.CascadeClassifier('C:/Users/kompiuteris/Desktop/python/room controlas/haarcascade_frontalface_default.xml')
    facemask_model=load_model("C:/Users/kompiuteris/Desktop/python/room controlas/model.h5")
    counter = 0
    while True:
        counter = 30
        has_mask = 0
        (rval, im)= webcam.read()
        # print(rval)

        if counter%30==0:

            im=cv2.flip(im,1,1) #Flip to act as a mirror
                
            try:
                # Resize the image to speed up detection
                mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))
            except:
                webcam.release()
                cv2.destroyAllWindows()
                break
            # detect MultiScale / faces 
            faces = classifier.detectMultiScale(mini)

            # Draw rectangles around each face
            for f in faces:
                (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
                #Save just the rectangle faces in SubRecFaces
                face_img = im[y:y+h, x:x+w]
                resized=cv2.resize(face_img,(input_shape,input_shape))
                normalized=resized/255.0
                reshaped=np.reshape(normalized,(1,input_shape, input_shape,3))
                reshaped = np.vstack([reshaped])
                result=facemask_model.predict(reshaped)
                # print(result)
                
                label=np.argmax(result,axis=1)[0]
                has_mask = bool(label)
                if queue_facemask.empty():
                    queue_facemask.put(has_mask)

                
                cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[label],2)
                cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[label],-1)
                cv2.putText(im, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
                # Show the image
            cv2.imshow('LIVE',   im)
            key = cv2.waitKey(1)
            if key == 27:
                webcam.release()
                cv2.destroyAllWindows()
                break
            

            #t_end = 

            # print(has_mask)
            #return has_mask
        counter += 1
           

def print_value():
    if not queue_aptikimas.empty():
        print('aptikimas', queue_aptikimas.get_nowait())

    if not queue_facemask.empty():
        print('facemask', queue_facemask.get_nowait()) 
    window.after(1000, print_value)

window = tk.Tk()
window.title("Aptikimo_sistema")
window.configure(background ='white')
window.grid_rowconfigure(0, weight = 1)
window.grid_columnconfigure(0, weight = 1)
message = tk.Label(
     window, text ="Atpažinimo sistema",
     bg ="green", fg = "white", width = 27,
     height = 3, font = ('times', 30, 'bold'))    
message.place(x = 200, y = 20)

import queue
queue_facemask = queue.Queue()
queue_aptikimas = queue.Queue()
b1 = Button(window, text ="Žmonių kiekis patalpoje", 
            fg ="white", bg ="green", 
            command= lambda: threading.Thread(target=aptikimas).start(),
            width = 20, height = 3,
            font =('times', 15, ' bold '))
b1.place(x = 200 , y=200)
b2 = Button(window, text ="Aptikti kaukę", 
            fg ="white", bg ="green", 
            command= lambda:  threading.Thread(target=facemask).start(),
            width = 20, height = 3,
            font =('times', 15, ' bold '))
b2.place(x = 600 , y=200)

print_value()
window.mainloop()
