'''This code is for running the model on streamlit as a web application.
please click on the link given in readme file to watch it work'''

import cv2
import numpy as np
from keras.models import load_model
import streamlit as st
from streamlit_webrtc import ClientSettings,VideoTransformerBase,WebRtcMode,webrtc_streamer


st.title('Real Time Face emotion recognition')

model=load_model('model.h5')

WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
        
        class_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']


        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detect.detectMultiScale(img_gray, 1.3,1)
        if faces is ():
            return img

        for(x,y,w,h) in faces:
            x = x - 5
            w = w + 10
            y = y + 7
            h = h + 2
            
            cv2.rectangle(img, (x,y+10),(x+w,y+h),(0,0,255), 3)
            img_crop = img[y:y+h,x:x+w]
            img_crop = img[y:y+h,x:x+w]                        
            final_image = cv2.resize(img_crop, (48,48))
            final_image = np.expand_dims(final_image, axis = 0)
            final_image = final_image/255.0
            prediction = model.predict(final_image)
            label=class_labels[prediction.argmax()]
            cv2.putText(img,label,(x,y), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)      
        return img

webrtc_streamer(
        key="emotion-detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_transformer_factory=VideoTransformer,
        async_transform=True,
    )
