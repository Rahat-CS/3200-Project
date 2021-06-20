#!/usr/bin/env python
# coding: utf-8

# In[2]:
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from imutils.video import VideoStream
import imutils
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image


top = tk.Tk()
top.geometry('1500x900')
top.title('Face Mask Detection')
top.configure(background='#CDCDCD')

label = Label(top, background='#CDCDCD', font=('arial', 18, 'bold'))
sign_image = Label(top)

prototxtPath=os.path.sep.join([r'D:\Face Mask Detection\face-mask-detector','deploy.prototxt'])
weightsPath=os.path.sep.join([r'D:\Face Mask Detection\face-mask-detector','res10_300x300_ssd_iter_140000.caffemodel'])



net=cv2.dnn.readNet(prototxtPath,weightsPath)


model=load_model(r'D:\Face Mask Detection\dataset')

#image=cv2.imread(r'D:\Face Mask Detection\examples\example_02.png')

def classifyImage(fp):
    image = cv2.imread(fp)
    (h,w)=image.shape[:2]


    blob=cv2.dnn.blobFromImage(image,1.0,(300,300),(104.0,177.0,123.0))



    net.setInput(blob)
    detections=net.forward()


    #loop over the detections
    for i in range(0,detections.shape[2]):
        confidence=detections[0,0,i,2]


        if confidence>0.5:
            #we need the X,Y coordinates
            box=detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX,startY,endX,endY)=box.astype('int')

            #ensure the bounding boxes fall within the dimensions of the frame
            (startX,startY)=(max(0,startX),max(0,startY))
            (endX,endY)=(min(w-1,endX), min(h-1,endY))


            #extract the face ROI, convert it from BGR to RGB channel, resize it to 224,224 and preprocess it
            face=image[startY:endY, startX:endX]
            face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
            face=cv2.resize(face,(224,224))
            face=img_to_array(face)
            face=preprocess_input(face)
            face=np.expand_dims(face,axis=0)

            (mask,withoutMask)=model.predict(face)[0]

            #determine the class label and color we will use to draw the bounding box and text
            label='Mask' if mask>withoutMask else 'No Mask'
            color=(0,255,0) if label=='Mask' else (0,0,255)

            #include the probability in the label
            label="{}: {:.2f}%".format(label,max(mask,withoutMask)*100)

            #display the label and bounding boxes
            cv2.putText(image,label,(startX,startY-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)
            cv2.rectangle(image,(startX,startY),(endX,endY),color,2)
            cv2.imshow("OutPut", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        
        
#cv2.imshow("OutPut",image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Image", command=lambda: classifyImage(file_path), padx=10, pady=5)
    classify_b.configure(background='#364196', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.6, rely=0.4)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass


def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations and list of predictions

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            # we need the X,Y coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')

            # ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel, resize it to 224,224 and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

        # only make a predictions if atleast one face was detected
        if len(faces) > 0:
            faces = np.array(faces, dtype='float32')
            preds = maskNet.predict(faces, batch_size=12)

        return (locs, preds)

def classifyVideo():
    prototxtPath = os.path.sep.join([r'D:\Face Mask Detection\face-mask-detector', 'deploy.prototxt'])
    weightsPath = os.path.sep.join(
        [r'D:\Face Mask Detection\face-mask-detector', 'res10_300x300_ssd_iter_140000.caffemodel'])

    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    maskNet = load_model(r'D:\Face Mask Detection\mobilenet_v2.model')

    vs = VideoStream(src=0).start()

    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # detect faces in the frame and preict if they are waring masks or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over the detected face locations and their corrosponding loactions

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we will use to draw the bounding box and text
            label = 'Mask' if mask > withoutMask else 'No Mask'
            color = (0, 255, 0) if label == 'Mask' else (0, 0, 255)

            # display the label and bounding boxes
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    vs.stop()

upload = Button(top, text="Upload a Image", command=upload_image, padx=10, pady=5)
upload.configure(background='#364196', foreground='white', font=('arial', 10, 'bold'))
#upload.pack(side=RIGHT, pady=70)
upload.place( relx=0.25, rely= 0.75)

upload1 = Button(top, text="Detect with Video", command=classifyVideo, padx=10, pady=5)
upload1.configure(background='#364196', foreground='white', font=('arial', 10, 'bold'))
#upload1.pack(side=LEFT, pady=70)
upload1.place( relx=0.75, rely= 0.75)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Face Mask Detection", pady=40, font=('Times New Roman', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364196')
heading.pack()
top.mainloop()


