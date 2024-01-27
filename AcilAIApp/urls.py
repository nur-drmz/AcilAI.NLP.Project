from django.urls import path
from . import views

urlpatterns = [
    path('', views.predictor, name = 'predictor'),
    path('predictor2/', views.predictor2, name='predictor2'),
]