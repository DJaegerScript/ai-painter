import cv2
import mediapipe as mp

class HandTracker():
    def __init__(self, mode=False, max_hands=2, model_complexity=1 ,detection_con=0.5, track_con=0.75):
        # parameter untuk deteksi hand
        self.mode = mode
        self.max_hands = max_hands
        self.modelComplex = model_complexity
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_hands = mp.solutions.hands 
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.modelComplex, self.detection_con, self.track_con) # deteksi hand
        self.mp_draw = mp.solutions.drawing_utils 
        self.tipIds = [4, 8, 12, 16, 20]
        

    # nemuin tangan dan gambar line di tangan
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # ubah ke RGB
        self.results = self.hands.process(imgRGB) # result dari posisi tangan dll

        if self.results.multi_hand_landmarks: 
            for handLms in self.results.multi_hand_landmarks: # handLms -> berapa tangan yg kedetect
                if draw:
                    self.mp_draw.draw_landmarks(img, handLms,
                    self.mp_hands.HAND_CONNECTIONS) # gambar dot and line di tangan

        return img

    # cari posisi tangan dan gambar kotak 
    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h) # posisi tangan sumbu x y 
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy]) # append posisi tangan
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                (0, 255, 0), 2) # gambar kotak di tangan

        return self.lmList

    # cek apakah jari naik
    def fingersUp(self):
        fingers = []
        # Thumb yg naik
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers yg naik
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers
