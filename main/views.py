from django.shortcuts import render
from django.views.decorators import gzip
from django.http.response import HttpResponse, StreamingHttpResponse

from main.face_detector import FaceDetector
from main.hand_tracker import HandTracker


# Create your views here.
@gzip.gzip_page
def index(request):
    hand_tracker = HandTracker(detection_con=0.65,max_hands=1)
    face_detector = FaceDetector(hand_tracker)
    return StreamingHttpResponse(face_detector.generateCam(), content_type='multipart/x-mixed-replace;boundary=frame')