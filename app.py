import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from classes import HandDetector

st.title("Virtual Whiteboard")

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.detector = HandDetector()
        self.canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        self.prevX, self.prevY = 0, 0
        self.current_color = (255, 255, 255)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        self.detector.findHands(img)
        lmList = self.detector.findPosition(img)
        
        if len(lmList) != 0:
            fingers = self.detector.countFingers(lmList)
            curr_x, curr_y = self.detector.getSmoothedPosition(lmList[8][0], lmList[8][1])
            
            if self.prevX == 0 and self.prevY == 0:
                self.prevX, self.prevY = curr_x, curr_y
            
            if fingers == 1:
                cv2.line(self.canvas, (self.prevX, self.prevY), (curr_x, curr_y), self.current_color, 10)
            elif fingers == 2:
                cv2.circle(self.canvas, (curr_x, curr_y), 30, (0, 0, 0), cv2.FILLED)
            
            self.prevX, self.prevY = curr_x, curr_y
        else:
            self.prevX, self.prevY = 0, 0

        # Combine
        combined = cv2.addWeighted(img, 0.5, self.canvas, 0.5, 0)
        return combined

webrtc_streamer(key="whiteboard", video_transformer_factory=VideoProcessor)