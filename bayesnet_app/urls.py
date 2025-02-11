from django.urls import path
from . import views

app_name = 'bayesnet_app'

urlpatterns = [
    path('', views.index, name='index'),
    path('login/', views.user_login, name='login'),
    path('register/', views.user_register, name='register'),
    path('logout/', views.user_logout, name='logout'),
    path('create/', views.create_network, name='create_network'),
    path('analyze/<str:network_id>/', views.analyze_network, name='analyze_network'),
    path('api/calculate-marginals/', views.calculate_marginals, name='calculate_marginals'),
    path('api/set-finding/', views.set_finding, name='set_finding'),
    path('api/plot-marginals/', views.plot_marginals, name='plot_marginals'),
    path('api/save-network/', views.save_network, name='save_network'),
    path('plot_network/', views.plot_network, name='plot_network'),
    path('api/save-network-db/', views.save_network_db, name='save_network_db'),
] 