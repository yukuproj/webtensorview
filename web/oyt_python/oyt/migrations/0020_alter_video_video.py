# Generated by Django 3.2 on 2022-12-14 10:52

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('oyt', '0019_auto_20221214_0910'),
    ]

    operations = [
        migrations.AlterField(
            model_name='video',
            name='video',
            field=models.FileField(max_length=2000, null=True, upload_to=''),
        ),
    ]
