# Generated by Django 3.2 on 2022-12-19 09:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('oyt', '0021_auto_20221218_1856'),
    ]

    operations = [
        migrations.AddField(
            model_name='video',
            name='transform_video_encoded',
            field=models.FileField(max_length=2000, null=True, upload_to=''),
        ),
    ]