# Generated by Django 3.2 on 2022-12-19 17:31

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('oyt', '0022_video_transform_video_encoded'),
    ]

    operations = [
        migrations.RenameField(
            model_name='video',
            old_name='transform_video_encoded',
            new_name='raw_video',
        ),
    ]
