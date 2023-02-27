from django.contrib import admin
from django.conf.urls import include
from django.urls import re_path
from django.views.generic import TemplateView

urlpatterns = [

    re_path('admin/', admin.site.urls),
    re_path("", include("PublicDataReader.urls")), 

]