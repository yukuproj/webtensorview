

"""oyt_python URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import include, path
from django.conf import settings
from django.contrib import admin
from django.urls import path
from oyt.views import HomeView
from oyt.views import NewVideoView
from oyt.views import NewVideoCompressedView
from oyt.views import LoginView
from oyt.views import RegisterView
from oyt.views import ErrorView
from oyt.views import VideoView
from oyt.views import CommentView
from oyt.views import LogoutView
from oyt.views import EditVideoView
from oyt.views import EditUserView
from oyt.views import CreatePlaylistView
from oyt.views import PlaylistIndexView
from oyt.views import PlaylistView
from oyt.views import AddVideoToPlaylistView
from oyt.views import PlaylistVideoView
from oyt.views import DeleteVideoView
from oyt.views import DownloadFileView
from oyt.views import TransformVideoView
from oyt.views import RemoveVideoView
from oyt.views import DeletePlaylistView
import debug_toolbar
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', HomeView.as_view()),
    path('new_video', NewVideoView.as_view()),
    path('new_video_compressed', NewVideoCompressedView.as_view()),
    path('login', LoginView.as_view()),
    path('register', RegisterView.as_view()),
    path('error', ErrorView.as_view()),
    path('video/<int:id>', VideoView.as_view()),
    path('comment', CommentView.as_view()),
    path('logout', LogoutView.as_view()),
    path('edit_video/<int:id>', EditVideoView.as_view()),
    path('edit_user', EditUserView.as_view()),
    path('new_playlist', CreatePlaylistView.as_view()),
    path('playlist_index', PlaylistIndexView.as_view()),
    path('add_to_playlist/<int:id>', AddVideoToPlaylistView.as_view()),
    path('playlist/<int:playlist_id>/', PlaylistView.as_view()),
    path('playlist/<int:playlist_id>/<int:video_id>',
         PlaylistVideoView.as_view()),
    path('delete_video/<int:id>/', DeleteVideoView.as_view()),
    path('download_video/<int:id>/', DownloadFileView.as_view()),
    path('transform_video/<int:id>/', TransformVideoView.as_view()),    
    
    path('remove_from_playlist/<int:id>/', RemoveVideoView.as_view()),
    path('delete_playlist/<int:id>/', DeletePlaylistView.as_view()),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

if settings.DEBUG:
    urlpatterns.append(path('__debug__/', include(debug_toolbar.urls)))
