# Generated by Django 4.2.7 on 2024-05-22 12:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('sign', '0004_alter_userprofile_lessons_completed'),
    ]

    operations = [
        migrations.AddField(
            model_name='lesson',
            name='video_file',
            field=models.FileField(default=1, upload_to='videos/'),
            preserve_default=False,
        ),
    ]
