from django.conf.urls import include
from django.urls import re_path
from .views import obj_detect_api
from django.views.generic import TemplateView

urlpatterns = [

    # 이미지 인식 버튼 - 결과 이미지 전송
    re_path("object_detect/", obj_detect_api),

]