from django.shortcuts import render
from django.views.decorators import gzip
from django.http.response import HttpResponse, StreamingHttpResponse

from main.painter import Painter
from main.hand_tracker import HandTracker


# Create your views here.
def index(request):
    return render(request, 'index.html')
    
@gzip.gzip_page
def video(request):
    hand_tracker = HandTracker(detection_con=0.5,max_hands=1,track_con=0.5)
    painter = Painter(hand_tracker)
    return StreamingHttpResponse(painter.generateCam(), content_type='multipart/x-mixed-replace;boundary=frame')