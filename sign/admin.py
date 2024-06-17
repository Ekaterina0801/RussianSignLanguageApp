from django.contrib import admin

from sign.models import *
# Register your models here.
admin.site.register(Category)
admin.site.register(Label)
admin.site.register(Lesson)
admin.site.register(UserProfile)


