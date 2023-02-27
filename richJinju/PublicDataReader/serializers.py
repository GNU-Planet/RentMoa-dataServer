# api/serializers.py

from rest_framework import serializers
from .models import DetachedhouseRent


class AllItemSerializer(serializers.ModelSerializer):
    class Meta:
        model = DetachedhouseRent
        fields = ('dong', 'buildYear', 'dealBuildingArea')