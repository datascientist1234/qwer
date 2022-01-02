'''
This python script contains code to create a stream lit app. This app detects facial emotions end to end.

To run this file "locally", type 
                    
                    streamlit run app.py --server.maxUploadSize=50
                    
Files required to run this app:
1. app.py
2. haarcascade_frontalface_default.xml
3. model.h5
4. demo_images    (optional) if you want to see demo images
'''

# importing relevant libraries
#import os
#import av
import cv2
#import time
#import tempfile
import numpy as np
#from PIL import Image
import streamlit as st
#from time import sleep
#from aiortc.contrib.media import MediaPlayer
from tensorflow.keras.models import load_model
#from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import (
    ClientSettings,
    VideoTransformerBase,
    WebRtcMode,
    webrtc_streamer,
)


WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

# importing the necessary files
faceCascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
model =load_model(r'model.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']


# a class that captures real time webcam feed
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)                               
        faces = fc.detectMultiScale(gray)                              

        for (x,y,w,h) in faces:
           cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)                     
           img_crop = gray[y:y+h,x:x+w]                                            
           img_crop = cv2.resize(img_crop,(48,48))    

        if faces is ():
            print("no faces detected")
            
            
        else:
            final_img = img_crop.astype('float')/255.0                              
            final_img = img_to_array(final_img)                                             
            final_img = np.expand_dims(final_img,axis=0)                                   

            prediction = model.predict(final_img)[0]                             
            label=emotions[prediction.argmax()]                           
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),3)  
         return img   
                    
  '''      
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray)
        if faces is ():
            return img

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = model.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y-10)
            cv2.putText(img,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)    
        return img


'''

# main function
webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_transformer_factory=VideoTransformer,
        async_transform=True,
    )
        
WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": True},
)

