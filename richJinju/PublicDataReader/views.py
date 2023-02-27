# api/views.py
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .models import DetachedhouseRent
from .serializers import AllItemSerializer

@api_view(['GET'])
def obj_detect_api(request):
    item = DetachedhouseRent.objects.get(id=2)
    serializer = AllItemSerializer(item)
    return Response(serializer.data)