from django.db import models
from django.contrib.auth.models import User


class Category(models.Model):
    name = models.CharField(max_length=100)
    picture_file = models.FileField(upload_to='category_pictures/', null=True)
    def __str__(self):
        return self.name

class Label(models.Model):
    name = models.CharField(max_length=100)
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
    def __str__(self):
        return self.name

class Lesson(models.Model):
    video_url = models.CharField(max_length=50)
    video_file = models.FileField(upload_to='videos/')
    label = models.ForeignKey(Label, on_delete=models.CASCADE)
    def __str__(self):
        return self.label.name

""" class Test(models.Model):
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
    def __str__(self):
        return self.category.name """


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    labels_learned = models.ManyToManyField(Label, null=True, related_name='learned_labels')
    tests_completed = models.ManyToManyField(Label, null=True, related_name='completed_tests')
    def __str__(self):
        return self.user.username



