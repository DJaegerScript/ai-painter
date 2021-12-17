import cv2
import threading
import numpy as np
import urllib.request as req

class Painter(object):
    def __init__(self, hand_tracker):
        self.req = req
        
        self.video = cv2.VideoCapture(0)
        self.video.set(3, 1280)
        self.video.set(4, 720)
        
        self.hand_tracker = hand_tracker
        
        self.draw_color = (255, 0, 255)
        self.header = self.__get_header(0)
        
        self.xp, self.yp = 0, 0
        self.img_canvas = np.zeros((720, 1280, 3), np.uint8)
        
        self.brushThickness = 15
        self.eraserThickness = 100
        
        (self.grabbed, self.frame) = self.video.read()
        
        threading.Thread(target=self.update).start()
    
    def __del__(self):
        self.video.release()
        
    def __track_fingers(self, lm_list, image):
        x1, y1 = lm_list[8][1:]
        x2, y2 = lm_list[12][1:]

        # 3. Check which fingers are up
        fingers = self.hand_tracker.fingersUp()

        # 4. If Selection Mode - Two finger are up (pilih mau warna apa / eraser)
        if fingers[1] and fingers[2]:
            self.xp, self.yp = 0, 0
            if y1 < 125:
                if 250 < x1 < 450:
                    self.header = self.__get_header(0)
                    self.draw_color = (255, 0, 255)
                elif 550 < x1 < 750:
                    self.header = self.__get_header(1)
                    self.draw_color = (255, 0, 0)
                elif 800 < x1 < 950:
                    self.header = self.__get_header(2)
                    self.draw_color = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    self.header = self.__get_header(3)
                    self.draw_color = (0, 0, 0)
            cv2.rectangle(image, (x1, y1 - 25), (x2, y2 + 25), self.draw_color, cv2.FILLED)

        # 5. If Drawing Mode - Index finger is up (ini untuk ngegambar)
        if fingers[1] and fingers[2] == False:
            cv2.circle(image, (x1, y1), 15, self.draw_color, cv2.FILLED)
            if self.xp == 0 and self.yp == 0:
                self.xp, self.yp = x1, y1

            cv2.line(image, (self.xp, self.yp), (x1, y1), self.draw_color, self.brushThickness)

            if self.draw_color == (0, 0, 0):
                cv2.line(image, (self.xp, self.yp), (x1, y1), self.draw_color, self.eraserThickness)
                cv2.line(self.img_canvas, (self.xp, self.yp), (x1, y1), self.draw_color, self.eraserThickness)
            
            else:
                cv2.line(image, (self.xp, self.yp), (x1, y1), self.draw_color, self.brushThickness)
                cv2.line(self.img_canvas, (self.xp, self.yp), (x1, y1), self.draw_color, self.brushThickness)

            self.xp, self.yp = x1, y1

        # Clear Canvas when all fingers are up
        if all (x >= 1 for x in fingers[:-1]):
            self.img_canvas = np.zeros((720, 1280, 3), np.uint8)
        
    def __get_frame(self):  
        image = self.frame
        image = cv2.flip(image, 1)
        image = self.hand_tracker.findHands(image)
        lm_list = self.hand_tracker.findPosition(image)
        
        if len(lm_list) != 0:
            self.__track_fingers(lm_list, image)       
        
        imgGray = cv2.cvtColor(self.img_canvas, cv2.COLOR_RGB2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2RGB)
        image = cv2.bitwise_and(image,imgInv)
        image = cv2.bitwise_or(image,self.img_canvas)
            
        image[0:125, 0:1280] = self.header
        image = cv2.addWeighted(image,1,self.img_canvas,0.6,0)
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
    
    def __get_header(self, index):
        image_urls = ['https://ai-painter.s3.ap-southeast-1.amazonaws.com/531305.jpg', 
                      'https://ai-painter.s3.ap-southeast-1.amazonaws.com/531306.jpg',
                      'https://ai-painter.s3.ap-southeast-1.amazonaws.com/531307.jpg',
                      'https://ai-painter.s3.ap-southeast-1.amazonaws.com/531308.jpg'] 
        
        overlay_list = []

        
        for image_url in image_urls:
            results = self.req.urlopen(image_url)
            img = np.asarray(bytearray(results.read()), dtype="uint8")
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            overlay_list.append(img)

        return overlay_list[index]        
    
    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()         
    
    def generateCam(self):
        while True:
            frame = self.__get_frame()
            yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')