# Generated by Django 3.2 on 2022-12-18 18:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('oyt', '0020_alter_video_video'),
    ]

    operations = [
        migrations.AddField(
            model_name='video',
            name='transform_chunk_size',
            field=models.IntegerField(null=True),
        ),
        migrations.AddField(
            model_name='video',
            name='transform_name',
            field=models.CharField(max_length=1000, null=True),
        ),
        migrations.AddField(
            model_name='video',
            name='transform_quality',
            field=models.IntegerField(null=True),
        ),
    ]
