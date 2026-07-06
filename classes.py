import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

class HandDetector:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options, 
            num_hands=1,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.results = None
        self.smooth_x, self.smooth_y = 0, 0
        self.alpha = 0.3 

    def findHands(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        self.results = self.detector.detect(mp_image)
        return img

    def findPosition(self, img):
        lmList = []
        if self.results and self.results.hand_landmarks:
            for hand_landmarks in self.results.hand_landmarks:
                for lm in hand_landmarks:
                    h, w, _ = img.shape
                    lmList.append([int(lm.x * w), int(lm.y * h)])
        return lmList

    def countFingers(self, lmList):
        if not lmList: return 0
        fingers = 0
        if lmList[8][1] < lmList[6][1]: fingers += 1 # Index
        if lmList[12][1] < lmList[10][1]: fingers += 1 # Middle
        return fingers

    def getSmoothedPosition(self, cx, cy):
        self.smooth_x = self.smooth_x + self.alpha * (cx - self.smooth_x)
        self.smooth_y = self.smooth_y + self.alpha * (cy - self.smooth_y)
        return int(self.smooth_x), int(self.smooth_y)