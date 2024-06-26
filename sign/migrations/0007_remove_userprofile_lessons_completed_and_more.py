# Generated by Django 4.2.13 on 2024-06-05 15:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('sign', '0006_userprofile_tests_completed_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='userprofile',
            name='labels_learned',
            field=models.ManyToManyField(null=True, related_name='learned_labels', to='sign.label'),
        ),
        migrations.RemoveField(
            model_name='userprofile',
            name='lessons_completed',
        ),
        migrations.AlterField(
            model_name='userprofile',
            name='tests_completed',
            field=models.ManyToManyField(null=True, related_name='completed_tests', to='sign.label'),
        ),
    ]
