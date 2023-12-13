from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path("admin/", admin.site.urls),
    # path("", views.home, name="home"),
    path("", views.home, name="index"),
    path("results/", views.results, name="results"),
]
