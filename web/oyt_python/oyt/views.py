

from django.core.files import File
from django.http import FileResponse
from django.shortcuts import render
from django.views.generic.base import View, HttpResponse, HttpResponseRedirect
from .forms import LoginForm, RegisterForm, NewVideoForm, CommentForm, EditVideoForm, EditUserForm, NewPlaylistForm, NewTransformForm, NewVideoCompressedForm
from django.contrib.auth.models import User
from django.core.exceptions import ObjectDoesNotExist
from django.contrib.auth import login, authenticate, logout
from django.db.models import Q
from django.urls import reverse
from .models import Video, Comment, Playlist
from hashlib import sha256
import string
import random
import time
import os
import subprocess

from TensorVision.cvtcomp.iocomp.io import load_video_to_numpy
import numpy as np
import matplotlib.pyplot as plt 
from TensorVision.cvtcomp.base.base import TensorVideo
from TensorVision.cvtcomp.iocomp.io import load_video_to_numpy, save_compressed_video, save_video_from_numpy, load_compressed_video
from TensorVision.cvtcomp.utils import play_video
from cv2 import PSNR

class LogoutView(View):
    def get(self, request):
        if request.user.is_authenticated:
            logout(request)
        return HttpResponseRedirect('/')


class HomeView(View):
    template_name = 'index.html'

    # Fetch videos from db
    # Fetch only public videos or private videos owned by user
    def get(self, request):
        most_recent_videos = Video.objects.order_by(
            '-datetime').filter(Q(is_private=False) | Q(user_id=request.user.id))[:100]
        most_liked_videos = Video.objects.order_by(
            '-num_likes').filter(Q(is_private=False) | Q(user_id=request.user.id))[:100]
        return render(request, self.template_name, {'most_recent_videos': most_recent_videos, 'most_liked_videos': most_liked_videos})

    # Get string searched for and filter videos with
    # string in name or description
    def post(self, request):
        name = request.POST.get('search_value')
        videos = Video.objects.filter(Q(is_private=False) | Q(
            user_id=request.user.id)).filter(Q(title__contains=name) | Q(description__contains=name))
        return render(request, self.template_name, {'searched': True, 'videos': videos})


class PlaylistIndexView(View):
    template_name = 'playlist_index.html'

    # Fetch videos from db
    # Fetch only public videos or private videos owned by user
    def get(self, request):
        most_recent_playlists = Playlist.objects.filter(
            Q(is_private=False) | Q(user_id=request.user.id)).order_by('name')[:10]
        return render(request, self.template_name, {'playlists': most_recent_playlists})

    # Get string searched for and filter playlists with
    # string in name or description
    def post(self, request):
        name = request.POST.get('search_value')
        playlists = Playlist.objects.filter(Q(is_private=False) | Q(
            user_id=request.user.id)).filter(Q(name__contains=name) | Q(description__contains=name))
        return render(request, self.template_name, {'playlists': playlists})


class VideoView(View):
    template_name = "video.html"

    def get(self, request, id):
        '''
        Get requested video by ID
        '''

        # Throw exception if video does not exist
        try:
            video_by_id = Video.objects.get(id=id)
        except ObjectDoesNotExist:
            return render(request, "error.html", {'error': "Error: Invalid Video URL. Video does not exist!"})


         
        # dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) 
        # transform_video_encoded = load_compressed_video(dir_path + "/media/" +video_by_id.transform_video_encoded.name)   
        # tensor_video = TensorVideo(compression_type= video_by_id.transform_name, quality=video_by_id.transform_quality, chunk_size=video_by_id.transform_chunk_size, decoded_data_type=np.uint8)
        # tensor_video.encoded_data = transform_video_encoded
        # restored_video = tensor_video.decode()
        # html = play_video(restored_video, fps=40 / 2)
        
        context = {
            "video": video_by_id,
            "video_type":  video_by_id.path.split(".")[-1],
            "liked": False,
           # "html" : html
        }

        # Handle case if video is private and not owned by user
        if video_by_id.is_private and request.user.id != video_by_id.user_id:
            return render(request, "error.html", {'error': "Error: Invalid video URL. video does not exist!"})

        # Compute likes
        liked_ids = video_by_id.likes
        if request.user.is_authenticated:
            if request.user.id in liked_ids:
                context['liked'] = True

        context['num_likes'] = video_by_id.num_likes

        # Display comment box only if user is authenticaed
        if request.user.is_authenticated == True:
            comment_form = CommentForm()
            context['form'] = comment_form

        # Display comments for video
        comments = Comment.objects.filter(
            video__id=id).order_by('-datetime')[:5]
        context['comments'] = comments

        return render(request, self.template_name, context)

    def post(self, request, id):
        '''
        Like/unlike specified video
        '''

        # Throw exception if video does not exist
        try:
            video_by_id = Video.objects.get(id=id)
        except ObjectDoesNotExist:
            return render(request, "error.html", {'error': "Error: Invalid Video URL. Video does not exist!"})
        like = request.POST['like']

        # Add/remove userID to liked IDs
        if like == 'True':
            if request.user.id not in video_by_id.likes:
                video_by_id.likes.append(request.user.id)
        else:
            video_by_id.likes.remove(request.user.id)

        # Compute number of likes
        video_by_id.num_likes = len(video_by_id.likes)
        video_by_id.save()

        return HttpResponseRedirect('/video/{}'.format(id))


class PlaylistView(View):
    template_name = "playlist.html"

    def get(self, request, playlist_id):
        '''
        Display videos inside playlist
        '''

        # Throw exception if playlist does not exist
        try:
            playlist_by_id = Playlist.objects.get(id=playlist_id)
        except ObjectDoesNotExist:
            return render(request, "error.html", {'error': "Error: Invalid Playlist URL. Playlist does not exist!"})

        # Handle case when playlist is private and not owned by user
        if playlist_by_id.is_private and request.user.id != playlist_by_id.user_id:
            return render(request, "error.html", {'error': "Error: Invalid Playlist URL. Playlist does not exist!"})

        video_ids = playlist_by_id.video_ids
        videos = Video.objects.filter(id__in=video_ids)
        context = {'videos': videos, 'playlist': playlist_by_id}

        return render(request, self.template_name, context)


class PlaylistVideoView(View):
    template_name = "playlist_video.html"

    def get(self, request, playlist_id, video_id):
        '''
        Display video within playlist along with list of video
        '''

        # Throw exception if playlist does not exist
        try:
            playlist_by_id = Playlist.objects.get(id=playlist_id)
        except ObjectDoesNotExist:
            return render(request, "error.html", {'error': "Error: Invalid Playlist URL. Playlist does not exist!"})

        # Handle case when playlist is private and not owned by user
        if playlist_by_id.is_private and request.user.id != playlist_by_id.user_id:
            return render(request, "error.html", {'error': "Error: Invalid Playlist URL. Playlist does not exist!"})

        # fetch video IDs and videos
        video_ids = playlist_by_id.video_ids
        videos = Video.objects.filter(id__in=video_ids)

        video_by_id = Video.objects.get(id=video_id)
        context = {
            'video': video_by_id,
            'videos': videos,
            'playlist': playlist_by_id,
            'video_type': video_by_id.path.split(".")[-1]
        }

        return render(request, self.template_name, context)


class LoginView(View):
    template_name = "login.html"

    def get(self, request):
        '''
        Render login page, redirect to home if logged in
        '''
        if request.user.is_authenticated:
            logout(request)
            return HttpResponseRedirect('/login')
        form = LoginForm()

        return render(request, self.template_name, {'form': form})

    def post(self, request):
        '''
        Authenticarte user login and redirect to home page if valid
        '''

        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']

            user = authenticate(
                request=request,
                username=username,
                password=password
            )

            if user is not None:
                login(request, user)
                return HttpResponseRedirect('/')
            else:
                return render(request, "error.html", {'error': "Error: Invalid Credentials!"})

        return HttpResponseRedirect('/')


class CommentView(View):
    template_name = "comment.html"

    def post(self, request):
        '''
        Post comment on video if user is authenticated
        '''
        form = CommentForm(request.POST)
        if form.is_valid():
            # Create Comment
            comment = form.cleaned_data['text']
            video_id = request.POST['video']

            # Throw exception if video does not exist
            try:
                video = Video.objects.get(id=video_id)
            except ObjectDoesNotExist:
                return render(request, "error.html", {'error': "Error: Invalid Video URL. Video does not exist!"})

            # Handle case if video is private and not owned by user
            if video.is_private and request.user.id != video.user_id:
                return render(request, "error.html", {'error': "Error: Invalid video URL. video does not exist!"})

            new_comment = Comment(
                user=request.user,
                text=comment,
                video=video
            )

            new_comment.save()

            return HttpResponseRedirect('/video/{}'.format(str(video_id)))
        else:
            return render(request, "error.html", {'error': "Error: Inavlid Form Input!"})


class RegisterView(View):
    template_name = "register.html"

    def get(self, request):
        '''
        Render sign up page
        '''

        if request.user.is_authenticated:
            return HttpResponseRedirect('/')
        form = RegisterForm()
        return render(request, self.template_name, {'form': form})

    def post(self, request):
        '''
        Validate and register user
        '''

        form = RegisterForm(request.POST)
        if form.is_valid():
            # Create Account
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            email = form.cleaned_data['email']
            first_name = form.cleaned_data['first_name']
            last_name = form.cleaned_data['last_name']

            # Verify Unique User
            try:
                existing_user = User.objects.get(username=username)
                if existing_user is not None:
                    return render(request, "error.html", {'error': "Error: Username is already taken!"})

            except ObjectDoesNotExist:
                pass

            # Verify Unique Email ID
            try:
                existing_email = User.objects.get(email=email)
                if existing_email is not None:
                    return render(request, "error.html", {'error': "Error: Email ID is already in use!"})
            except ObjectDoesNotExist:
                pass

            # Create object
            new_user = User(
                username=username,
                email=email,
                first_name=first_name,
                last_name=last_name
            )

            # Hash and set password
            new_user.set_password(password)

            # Save user
            new_user.save()
            return HttpResponseRedirect('/login')
        else:
            return render(request, "error.html", {'error': "Error: Inavlid Form Input!"})


""" class NewVideoView(View):
    template_name = 'new_video.html'
    supported_types = ['video/mp4', 'video/webm']

    def get(self, request):
        '''
        Render upload video page, redirect to login if not signed in
        '''
        if request.user.is_authenticated == False:
            return HttpResponseRedirect('/login')
        form = NewVideoForm()

        return render(request, self.template_name, {'form': form})

    def post(self, request):
        '''
        Validate video upload input and save video
        '''

        form = NewVideoForm(request.POST, request.FILES)
        if form.is_valid():
            # create a new Video Entry
            title = form.cleaned_data['title']
            description = form.cleaned_data['description']
            video = form.cleaned_data['video']
            is_private = form.cleaned_data['is_private']

            # Verify video format EDITED
           # if video.content_type not in self.supported_types:
            #    return render(request, "error.html", {'error': "Error: Inavlid Video format {}!".format(video.content_type)})

            # Generate unique string for video
            hash_str = str(round(time.time() * 1000)) + \
                str(video.name) + str(request.user)

            hash = sha256(hash_str.encode())
            path = hash.hexdigest()[:10] + "_" + video.name
            video.name = path

            # Create and save video object
            new_video = Video(
                title=title,
                description=description,
                user=request.user,
                path="/media/" + path,
                video=video,
                is_private=is_private,
                likes=[]
            )
            new_video.save()

            # Generate thumbnail for video
            dir_path = os.path.dirname(
                os.path.dirname(os.path.realpath(__file__)))

            video_input_path = dir_path + '/media/' + path
            img_output_path = dir_path + '/media/' + path + '.jpg'
            os.system('ffmpeg -i {ip} -ss 00:00:00.000 -vframes 1 {op}'.format(
                ip=video_input_path, op=img_output_path))

            # redirect to detail view template of a Video
            return HttpResponseRedirect('/video/{id}'.format(id=new_video.id))
        else:
            return render(request, "error.html", {'error': "Error: Inavlid Form Input!"}) """

class NewVideoView(View):
    template_name = 'new_video.html'
    supported_types = ['video/mp4', 'video/webm']

    def get(self, request):
        '''
        Render upload video page, redirect to login if not signed in
        '''
        if request.user.is_authenticated == False:
            return HttpResponseRedirect('/login')
        form = NewVideoForm()

        return render(request, self.template_name, {'form': form})

    def post(self, request):
        '''
        Validate video upload input and save video
        '''

        form = NewVideoForm(request.POST, request.FILES)
        if form.is_valid():
            # create a new Video Entry
            title = form.cleaned_data['title']
            description = form.cleaned_data['description']
            video = form.cleaned_data['video']
            is_private = form.cleaned_data['is_private']

            # Verify video format EDITED
           # if video.content_type not in self.supported_types:
            #    return render(request, "error.html", {'error': "Error: Inavlid Video format {}!".format(video.content_type)})

            # Generate unique string for video
            hash_str = str(round(time.time() * 1000)) + \
                str(video.name) + str(request.user)

            hash = sha256(hash_str.encode())
            path = hash.hexdigest()[:10] + "_" + video.name
            video.name = path


            # Create and save video object
            new_video = Video(
                title=title,
                description=description,
                user=request.user,
                path="/media/" + path,
                video=video,
                is_private=is_private,
                likes=[]
            )
            new_video.save()

##
            dir_path = os.path.dirname(
                os.path.dirname(os.path.realpath(__file__)))
                
            os.system('ffmpeg -i {ip} {op}'.format(
                ip=dir_path + "/media/" + path,
                 op=dir_path + '/media/' +  "viewable_" + path + ".mp4"))
##
            
            new_video.video.name = "viewable_" + path + ".mp4"
            new_video.path = '/media/' +  "viewable_" + path + ".mp4"
            new_video.transform_video_encoded.name =  path
            new_video.save()
            # Generate thumbnail for video


            video_input_path = dir_path + '/media/' + "viewable_" + path + ".mp4"
            img_output_path = dir_path + '/media/' + "viewable_" + path + ".mp4" + '.jpg'
            os.system('ffmpeg -i {ip} -ss 00:00:00.000 -vframes 1 {op}'.format(
                ip=video_input_path, op=img_output_path))

            # redirect to detail view template of a Video
            return HttpResponseRedirect('/video/{id}'.format(id=new_video.id))
        else:
            return render(request, "error.html", {'error': "Error: Inavlid Form Input!"})


class NewVideoCompressedView(View):
    template_name = 'new_video_compressed.html'
    supported_types = ['video/mp4', 'video/webm']

    def get(self, request):
        '''
        Render upload video page, redirect to login if not signed in
        '''
        if request.user.is_authenticated == False:
            return HttpResponseRedirect('/login')
        form = NewVideoCompressedForm()

        return render(request, self.template_name, {'form': form})

    def post(self, request):
        '''
        Validate video upload input and save video
        '''

        form = NewVideoCompressedForm(request.POST, request.FILES)
        if form.is_valid():
            # create a new Video Entry
            title = form.cleaned_data['title']
            description = form.cleaned_data['description']
            video = form.cleaned_data['video']
            is_private = form.cleaned_data['is_private']

            # Verify video format EDITED
           # if video.content_type not in self.supported_types:
            #    return render(request, "error.html", {'error': "Error: Inavlid Video format {}!".format(video.content_type)})

            # Generate unique string for video
            hash_str = str(round(time.time() * 1000)) + \
                str(video.name) + str(request.user)

            hash = sha256(hash_str.encode())
            path = hash.hexdigest()[:10] + "_" + video.name
            video.name = path

            dir_path = os.path.dirname(
                os.path.dirname(os.path.realpath(__file__)))


            # Create and save video object
            new_video = Video(
                title=title,
                description=description,
                user=request.user,
                path="/media/" + path,       #pickle so far - has to be replaced
                video=video,                #pickle so far - has to be replaced
                is_private=is_private,
                likes=[]
            )
            new_video.save()


## restore             
            tensor_video = TensorVideo()
            restored_tensor_video = tensor_video.load(dir_path + "/media/" + path)              
            new_video.transform_name = restored_tensor_video.compression_type
            new_video.transform_quality =  restored_tensor_video.quality
            new_video.transform_chunk_size =  restored_tensor_video.chunk_size                     
            new_video.transform_video_encoded.name = dir_path + "/media/" + path   #pickle   
             
            name_avi =  "decoded_" + path + ".avi"
            restored_video = tensor_video.decode()                 
            save_video_from_numpy(dir_path + "/media/" + name_avi,
                                 restored_video,
                                fourcc =808596553, 
                                fps = restored_tensor_video.fps,
                                size= restored_tensor_video.encoded_data_size, 
                                color=True) 
##
                
            os.system('ffmpeg -i {ip} {op}'.format(
                ip=dir_path + "/media/" + name_avi,
                 op=dir_path + '/media/' +  "viewable_" + path + ".mp4"))
##
            
            new_video.video.name = "viewable_" + path + ".mp4"
            new_video.path = '/media/' +  "viewable_" + path + ".mp4"           
            new_video.save()
            # Generate thumbnail for video


            video_input_path = dir_path + '/media/' + "viewable_" + path + ".mp4"
            img_output_path = dir_path + '/media/' + "viewable_" + path + ".mp4" + '.jpg'
            os.system('ffmpeg -i {ip} -ss 00:00:00.000 -vframes 1 {op}'.format(
                ip=video_input_path, op=img_output_path))

            # redirect to detail view template of a Video
            return HttpResponseRedirect('/video/{id}'.format(id=new_video.id))
        else:
            return render(request, "error.html", {'error': "Error: Inavlid Form Input!"})



class TransformVideoView(View):
    template_name = 'transform_video.html'
    supported_types = ['video/mp4', 'video/webm']

    def get(self, request, id):
        '''
        Render upload video page, redirect to login if not signed in
        '''
        if request.user.is_authenticated == False:
            return HttpResponseRedirect('/login')
        form = NewTransformForm()

        return render(request, self.template_name, {'form': form})

    def post(self, request, id):
        '''
        Validate video upload input and save video
        '''
# Throw exception if video does not exist
        try:
            video_by_id = Video.objects.get(id=id)
        except ObjectDoesNotExist:
            return render(request, "error.html", {'error': "Error: Invalid Video URL. Video does not exist!"})

        context = {
            "video": video_by_id,
            "video_type": video_by_id.path.split(".")[-1],
            "liked": False
        }

        # Handle case if video is private and not owned by user
        if video_by_id.is_private and request.user.id != video_by_id.user_id:
            return render(request, "error.html", {'error': "Error: Invalid video URL. video does not exist!"})
 

        form = NewTransformForm(request.POST)
        if True: 
            # create a new Video Entry
            transform_name = form.data['transform_name']
            transform_quality = int(form.data['transform_quality'])
            transform_chunk_size = int(form.data['transform_chunk_size'])
             
            name_0 = video_by_id.transform_video_encoded.name            

            # Generate unique string for video
            hash_str = str(round(time.time() * 1000)) + str(name_0) + str(request.user)

            hash = sha256(hash_str.encode())
            name_1 = transform_name + "_" + str(transform_quality) +"_" + str(transform_chunk_size)+"_"+ hash.hexdigest()[:10] + "_" + name_0
            name_1 =name_1.replace('.','_')
            
            # Create and save video object
            dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) 

            #evoking new algorithm             
            video_original, fourcc, fps, size = load_video_to_numpy(dir_path + '/media/' + video_by_id.transform_video_encoded.name) 
       
            if transform_name=='tucker' or transform_name=='tt':               
                tensor_video = TensorVideo(compression_type= transform_name, quality=transform_quality, chunk_size=transform_chunk_size, decoded_data_type=np.uint8)
                video_encoded = tensor_video.encode(video_original, show_results=True)
                save_compressed_video(dir_path + "/media/" + name_1, video_encoded)

                name_avi =  "decoded_" + name_1 + ".avi"
                restored_video = tensor_video.decode()               
                save_video_from_numpy(dir_path + "/media/" + name_avi, restored_video, fourcc, fps, size, color=True) 

                name_pickled= "pickled_" + name_1
                tensor_video.save(dir_path + "/media/" + name_pickled)

            name_mp4 = "decoded_" + name_1 + ".mp4"
            os.system('ffmpeg -i {ip} {op}'.format(
                ip=dir_path + "/media/" + name_avi,
                 op=dir_path + "/media/" + name_mp4))
            
            file_stats_0 = os.stat(dir_path + '/media/' + video_by_id.transform_video_encoded.name)
            file_size_before = file_stats_0.st_size 
            file_stats_1 = os.stat(dir_path + "/media/" + name_1)
            file_size_after = file_stats_1.st_size 
            compression_ratio = file_size_before / file_size_after

            PSNR_val = PSNR(restored_video, video_original)


            new_video = Video(
                title= "compressed_" + transform_name + "_" + str(transform_quality) +"_" + str(transform_chunk_size)+"_" + video_by_id.title,
                description= str(transform_name) + " params: quality=" + str(transform_quality) +" ,chunk_size=" + str(transform_chunk_size)
                + '  ,compression_ratio=' + str(round(compression_ratio,2))+ '  ,PSNR=' + str(round(PSNR_val,2)) +' dB',
                user=request.user,
                path= '/media/' +  name_mp4,
                video= None,   
                is_private= video_by_id.is_private,
                likes=[],
                parent=None,
                info=None,
                transform_name = transform_name,
                transform_quality = transform_quality,
                transform_chunk_size = transform_chunk_size,
                transform_video_encoded = None,
                transform_compression_ratio = None
            )  
              
            new_video.video.name =  name_mp4
            new_video.transform_video_encoded.name = name_pickled # name_1          
            new_video.parent = video_by_id
            new_video.save()
             
            video_input_path = dir_path + '/media/' + name_mp4             
            img_output_path = dir_path + '/media/' +  name_mp4+ '.jpg'
            os.system('ffmpeg -i {ip} -ss 00:00:00.000 -vframes 1 {op}'.format(
                ip=video_input_path, op=img_output_path))           

            # redirect to detail view template of a Video
            return HttpResponseRedirect('/video/{id}'.format(id=new_video.id))
        else:
            return render(request, "error.html", {'error': "Error: Inavlid Form Input!"})

# class TransformVideoView(View):
#     template_name = 'transform_video.html'
#     supported_types = ['video/mp4', 'video/webm']

#     def get(self, request, id):
#         '''
#         Render upload video page, redirect to login if not signed in
#         '''
#         if request.user.is_authenticated == False:
#             return HttpResponseRedirect('/login')
#         form = NewTransformForm()

#         return render(request, self.template_name, {'form': form})

#     def post(self, request, id):
#         '''
#         Validate video upload input and save video
#         '''
# # Throw exception if video does not exist
#         try:
#             video_by_id = Video.objects.get(id=id)
#         except ObjectDoesNotExist:
#             return render(request, "error.html", {'error': "Error: Invalid Video URL. Video does not exist!"})

#         context = {
#             "video": video_by_id,
#             "video_type": video_by_id.path.split(".")[-1],
#             "liked": False
#         }

#         # Handle case if video is private and not owned by user
#         if video_by_id.is_private and request.user.id != video_by_id.user_id:
#             return render(request, "error.html", {'error': "Error: Invalid video URL. video does not exist!"})
 

#         form = NewTransformForm(request.POST)
#         if True: 
#             # create a new Video Entry
#             transform_name = form.data['transform_name']
#             transform_quality = form.data['transform_params'] 
#             transform_chumk_size = form.data['transform_chumk_size'] 
             
#             name_2 = video_by_id.video.name            

#             # Generate unique string for video
#             hash_str = str(round(time.time() * 1000)) + str(name_2) + str(request.user)

#             hash = sha256(hash_str.encode())
#             name_extended_2 = hash.hexdigest()[:10] + "_" + transform_name + "_" + name_2
         
#             filepath_2 = "/media/" + name_extended_2

#             # Create and save video object
#             new_video = Video(
#                 title= "transformed_" + video_by_id.title,
#                 description= "params_" + transform_quality,
#                 user=request.user,
#                 path= filepath_2,  
#                 video= None,   
#                 is_private= video_by_id.is_private,
#                 likes=[],
#                 parent=None,
#                 info=None
#             )
            
#             dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))    
#             video_2 = File(video_by_id.video)  
#             video_2.name = name_extended_2   
#             new_video.video.save(dir_path + filepath_2, video_2)
#             new_video.save()

#             # Generate thumbnail for video
#             video_input_path = dir_path + '/media/' + name_extended_2
#             img_output_path = dir_path + '/media/' + name_extended_2 + '.jpg'
#             os.system('ffmpeg -i {ip} -ss 00:00:00.000 -vframes 1 {op}'.format(
#                 ip=video_input_path, op=img_output_path))

#             # redirect to detail view template of a Video
#             return HttpResponseRedirect('/video/{id}'.format(id=new_video.id))
#         else:
#             return render(request, "error.html", {'error': "Error: Inavlid Form Input!"})

class DownloadFileView(View):
    template_name = "download_video.html"


    def get(self, request, id):
        '''
        Get requested video by ID
        '''

        # Throw exception if video does not exist
        try:
            video_by_id = Video.objects.get(id=id)
        except ObjectDoesNotExist:
            return render(request, "error.html", {'error': "Error: Invalid Video URL. Video does not exist!"})

        context = {
            "video": video_by_id,
            "video_type": video_by_id.path.split(".")[-1],
            "liked": False
        }

        # Handle case if video is private and not owned by user
        if video_by_id.is_private and request.user.id != video_by_id.user_id:
            return render(request, "error.html", {'error': "Error: Invalid video URL. video does not exist!"})

        # # Compute likes
        # liked_ids = video_by_id.likes
        # if request.user.is_authenticated:
        #     if request.user.id in liked_ids:
        #         context['liked'] = True

        # context['num_likes'] = video_by_id.num_likes

        # # Display comment box only if user is authenticaed
        # if request.user.is_authenticated == True:
        #     comment_form = CommentForm()
        #     context['form'] = comment_form

        # # Display comments for video
        # comments = Comment.objects.filter(
        #     video__id=id).order_by('-datetime')[:5]
        # context['comments'] = comments

        # return render(request, self.template_name, context)
        response = FileResponse(video_by_id.transform_video_encoded.file)
        # if video_by_id.transform_name is not None:
        #     if len(video_by_id.transform_name) > 0:
        #         response = FileResponse(video_by_id.transform_video_encoded.file)
        # else:
        #     response = FileResponse(video_by_id.video.file)
    

        #response = FileResponse(video_by_id.video.file)
        response['Content-Disposition'] = 'attachment; filename="'+os.path.basename(video_by_id.title)+'"'
        return response

        #return FileResponse(video_by_id.video.file, as_attachment=True, filename=video_by_id.title)



       # response = FileResponse(open(filename, 'rb'))
       # return response

    def post(self, request, id):
        # '''
        # Like/unlike specified video
        # '''

        # # Throw exception if video does not exist
        # try:
        #     video_by_id = Video.objects.get(id=id)
        # except ObjectDoesNotExist:
        #     return render(request, "error.html", {'error': "Error: Invalid Video URL. Video does not exist!"})
        # like = request.POST['like']

        # # Add/remove userID to liked IDs
        # if like == 'True':
        #     if request.user.id not in video_by_id.likes:
        #         video_by_id.likes.append(request.user.id)
        # else:
        #     video_by_id.likes.remove(request.user.id)

        # # Compute number of likes
        # video_by_id.num_likes = len(video_by_id.likes)
        # video_by_id.save()

        return HttpResponseRedirect('/video/{}'.format(id))


# def download(request, id):
#     obj = your_model_name.objects.get(id=id)
#     filename = obj.model_attribute_name.path
#     response = FileResponse(open(filename, 'rb'))
#     return response



class CreatePlaylistView(View):
    template_name = "new_playlist.html"

    def get(self, request):
        '''
        Render create playlist page, redirect to login if not signed in
        '''

        if request.user.is_authenticated == False:
            return HttpResponseRedirect('/login')
        form = NewPlaylistForm()
        return render(request, self.template_name, {'form': form})

    def post(self, request):
        '''
        Validate playlist creation input and save playlist
        '''

        form = NewPlaylistForm(request.POST, request.FILES)
        if form.is_valid():
            # create a new Playlist Entry
            name = form.cleaned_data['name']
            is_private = form.cleaned_data['is_private']
            description = form.cleaned_data['description']
            user = request.user

            new_playlist = Playlist(
                name=name,
                is_private=is_private,
                user=user,
                description=description,
                video_ids=[]
            )

            new_playlist.save()

            return render(request, "error.html", {'msg': "Playlist Created!"})
        else:
            return render(request, "error.html", {'error': "Error: Inavlid Form Input!"})


class AddVideoToPlaylistView(View):
    template_name = "add_to_playlist.html"

    def get(self, request, id):
        '''
        Render list of playlists owned by user
        '''

        # fetch playlists from db
        playlists = Playlist.objects.filter(
            user_id=request.user.id).order_by('name')
        video_id = id

        return render(request, self.template_name, {'playlists': playlists, 'video_id': video_id})

    def post(self, request, id):
        '''
        Add video to list of selected playlists
        '''

        playlists = request.POST.getlist('checks[]')
        video_id = id
        for playlist in playlists:
            playlist_id = int(playlist)
            playlist_obj = Playlist.objects.get(id=playlist_id)
            video_list = playlist_obj.video_ids
            if video_id not in video_list:
                video_list.append(video_id)
            playlist_obj.video_ids = video_list
            playlist_obj.save()

        return HttpResponseRedirect('/')


class EditVideoView(View):
    template_name = "edit_video.html"

    def get(self, request, id):
        '''
        Render edit video page
        '''

        if request.user.is_authenticated == False:
            return HttpResponseRedirect('/login')
        form = EditVideoForm()
        return render(request, self.template_name, {'form': form})

    def post(self, request, id):
        '''
        Validate input from edit video and modify metadata
        '''

        form = EditVideoForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                video_by_id = Video.objects.get(id=id)
            except ObjectDoesNotExist:
                return render(request, "error.html", {'error': "Error: Invalid Video URL. Video does not exist!"})

            if request.user.id != video_by_id.user_id:
                return render(request, "error.html", {'error': "Error: you are not the owner. You cannot modify this video!"})

            title = form.cleaned_data['title']
            description = form.cleaned_data['description']
            is_private = form.cleaned_data['is_private']

            if title != '':
                video_by_id.title = title

            if description != '':
                video_by_id.description = description

            if is_private == True or is_private == False:
                video_by_id.is_private = is_private

            video_by_id.save()
            return HttpResponseRedirect('/video/{id}'.format(id=video_by_id.id))
        else:
            return render(request, "error.html", {'error': "Error: Inavlid Form Input!"})


class EditUserView(View):
    template_name = "edit_user.html"

    def get(self, request):
        '''
        Render edit user page
        '''

        if request.user.is_authenticated == False:
            return HttpResponseRedirect('/login')
        form = EditUserForm()
        return render(request, self.template_name, {'form': form})

    def post(self, request):
        '''
        Validate input from edit user and modify entry
        '''

        form = EditUserForm(request.POST, request.FILES)
        id = request.user.id
        if form.is_valid():
            try:
                user_by_id = User.objects.get(id=id)
            except ObjectDoesNotExist:
                return render(request, "error.html", {'error': "Error: Invalid user ID. User does not exist!"})

            password = form.cleaned_data['password']
            first_name = form.cleaned_data['first_name']
            last_name = form.cleaned_data['last_name']

            if password != '':
                user_by_id.set_password(password)

            if first_name != '':
                user_by_id.first_name = first_name

            if last_name != '':
                user_by_id.last_name = last_name

            user_by_id.save()
            return HttpResponseRedirect('/')
        else:
            return render(request, "error.html", {'error': "Error: Inavlid Form Input!"})


class DeleteVideoView(View):
    template_name = "delete_video.html"

    def get(self, request, id):
        '''
        Render video deletion confirmation page
        '''

        video_by_id = Video.objects.get(id=id)

        return render(request, self.template_name, {'video': video_by_id})

    def post(self, request, id):
        '''
        Delete video and corresponding playlist entries
        '''

        try:
            video_by_id = Video.objects.get(id=id)
        except ObjectDoesNotExist:
            return render(request, "error.html", {'error': "Error: Invalid Video URL. Video does not exist!"})

        if request.user.id != video_by_id.user_id:
            return render(request, "error.html", {'error': "Error: you are not the owner. You cannot modify this video!"})

        playlists = Playlist.objects.all()
        for playlist in playlists:
            video_ids = playlist.video_ids
            try:
                video_ids.remove(id)
            except:
                continue
            playlist.save()

        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

        path = dir_path + video_by_id.path
        try:
            os.remove(path)
            os.remove(path + '.jpg')
        except:
            pass
        video_by_id.delete()

        return render(request, "error.html", {'msg': "Video Deleted!"})


class RemoveVideoView(View):
    template_name = "remove_from_playlist.html"

    def get(self, request, id):
        '''
        Render remove video from playlist page
        '''

        playlist_by_id = Playlist.objects.get(id=id)
        video_ids = playlist_by_id.video_ids
        videos = Video.objects.filter(id__in=video_ids)
        context = {'videos': videos, 'playlist': playlist_by_id}

        return render(request, self.template_name, context)

    def post(self, request, id):
        '''
        Validate list of videos to be removed and remove them
        '''

        videos = request.POST.getlist('checks[]')
        playlist_id = id
        playlist_obj = Playlist.objects.get(id=playlist_id)
        video_list = playlist_obj.video_ids
        for video in videos:
            video_id = int(video)
            try:
                video_list.remove(video_id)
            except:
                continue
        playlist_obj.video_ids = video_list
        playlist_obj.save()

        return HttpResponseRedirect('/playlist/{}'.format(playlist_id))


class DeletePlaylistView(View):
    template_name = "delete_playlist.html"

    def get(self, request, id):
        '''
        Render delete playlist page
        '''

        playlist_by_id = Playlist.objects.get(id=id)
        return render(request, self.template_name, {'playlist': playlist_by_id})

    def post(self, request, id):
        '''
        Validate and delete playlist
        '''

        try:
            playlist_by_id = Playlist.objects.get(id=id)
        except ObjectDoesNotExist:
            return render(request, "error.html", {'error': "Error: Invalid Playlist URL. Playlist does not exist!"})

        if request.user.id != playlist_by_id.user.id:
            return render(request, "error.html", {'error': "Error: you do not own this playlist. You cannot modify it!"})

        playlist_by_id.delete()

        return render(request, "error.html", {'msg': "Playlist Deleted!"})


class ErrorView(View):
    template_name = "error.html"
    error_string = "error"

    def get(self, request):
        '''
        Display error message
        '''
        return render(request, self.template_name, {'error': self.error_string})

    def setError(self, error_msg):
        self.error_string = error_msg
