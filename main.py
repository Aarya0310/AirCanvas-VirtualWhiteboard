import cv2
import numpy as np
from classes import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector()
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Palette + Clear Button Settings
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]
buttons = [(50, 20, 100, 70), (120, 20, 170, 70), (190, 20, 240, 70), (260, 20, 310, 70)]
clear_button = (330, 20, 400, 70) # New Clear Button coordinates
current_color = (255, 255, 255)
prevX, prevY = 0, 0

try:
    while True:
        success, img = cap.read()
        if not success: break
        img = cv2.flip(img, 1)
        
        # Draw UI
        for i, (x1, y1, x2, y2) in enumerate(buttons):
            cv2.rectangle(img, (x1, y1), (x2, y2), colors[i], cv2.FILLED)
        # Draw Clear Button
        cv2.rectangle(img, (clear_button[0], clear_button[1]), (clear_button[2], clear_button[3]), (0, 0, 255), cv2.FILLED)
        cv2.putText(img, "CLR", (clear_button[0]+10, clear_button[1]+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        detector.findHands(img)
        lmList = detector.findPosition(img)
        
        if len(lmList) != 0:
            fingers = detector.countFingers(lmList)
            x, y = detector.getSmoothedPosition(lmList[8][0], lmList[8][1])
            
            # Palette Selection
            for i, (x1, y1, x2, y2) in enumerate(buttons):
                if x1 < x < x2 and y1 < y < y2:
                    current_color = colors[i]
            
            # Clear Action
            if clear_button[0] < x < clear_button[2] and clear_button[1] < y < clear_button[3]:
                canvas = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Draw / Erase Logic
            if fingers == 1:
                if prevX == 0 and prevY == 0: prevX, prevY = x, y
                cv2.line(canvas, (prevX, prevY), (x, y), current_color, 10)
            elif fingers == 2:
                cv2.circle(canvas, (x, y), 30, (0, 0, 0), cv2.FILLED)
            
            prevX, prevY = x, y
        else:
            prevX, prevY = 0, 0

        combined = cv2.addWeighted(img, 0.5, canvas, 0.5, 0)
        cv2.imshow("Virtual Whiteboard", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    cap.release()
    if hasattr(detector, 'detector'): detector.detector.close()
    cv2.destroyAllWindows()