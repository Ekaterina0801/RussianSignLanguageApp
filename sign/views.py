import base64
from django.http import HttpResponse, HttpResponseRedirect, StreamingHttpResponse
from django.shortcuts import render, redirect
import cv2
from django.urls import reverse
import requests
from django.contrib.auth.models import User
from sign.models import *
from sign.processing import *
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from sign.constants import ORD2SIGN
import random
from django.http import JsonResponse
import json
from django.middleware import csrf
import logging
from urllib.parse import quote, urlencode

logger = logging.getLogger(__name__)


def signup(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            UserProfile.objects.create(user=user)
            return redirect('login')
    else:
        form = UserCreationForm()
    
    return render(request, 'registration/signup.html', {'form': form})

def index(request):
    return render (request, "index.html")

def recognizer(request):
    return render (request, "recognizer.html")

def checker(request,label):
    categorized_labels = getCategorizedLabels(None)
    return render (request, "checker.html", {'target_label_name':label, 'categorized_labels': categorized_labels})



def lessons_main(request):
    if request.user.is_authenticated:
        user_profile = UserProfile.objects.get(user=request.user)
    else:
        user_profile = None
    categorized_labels = getCategorizedLabels(user_profile)
    return render (request, "lessons_main.html", context={ 'categorized_labels': categorized_labels })

def tests_main(request):
    if request.user.is_authenticated:
        user_profile = UserProfile.objects.get(user=request.user)
    else:
        user_profile = None
    categorized_labels = getCategorizedLabels(user_profile)
    return render (request, "tests_main.html", context={ 'categorized_labels': categorized_labels })


def lesson(request,category):
    labels = getLabelsByCategory(category)
    try:
        if request.method == 'POST':
            label_name = request.POST.get('label')
            label = labels.get(name=label_name)
        else:
            label = labels[0]
    except:
        return redirect('/') # FIXME
    video_path = Lesson.objects.filter(label=label).first().video_file
    return render (request, "lesson.html", context={'labels':labels, 'target_label':label, 'video_path':video_path})


@login_required
def profile(request):
    user_profile = UserProfile.objects.get(user = request.user)
    categorized_labels = getCategorizedLabels(user_profile)
    return render (request, "profile.html",context={'categorized_labels':categorized_labels})


def start_test(request,category):
    labels = getLabelsByCategory(category)

    labels_names = [label.name for label in labels]
    labels_names.sort()

    types = ['show_gesture', 'guess_gesture']
    questions = []
    for label in labels:
        lesson = Lesson.objects.filter(label=label).first()
        questions.append({
            'id': label.id,
            'name': label.name,
            'video': str(lesson.video_file) if lesson.video_file else None,
            'type': random.choice(types)
        })
    random.shuffle(questions)

    request.session['questions'] = json.dumps(questions)
    request.session['answers'] = []
    request.session['lessons'] = json.dumps(labels_names)
    if 'correct_answers' in request.session:
        del request.session['correct_answers']
    request.session.modified = True
    request.session.save()

    redirect_url = f"{reverse('test', args=[1])}"
    print('redirect after start')
    return redirect(redirect_url)

def test(request, number):
    if 'correct_answers' in request.session:
        return redirect(f"{reverse('finish_test')}")
    
    if 'answers' not in request.session or 'questions' not in request.session:
        print('error not found')
        return HttpResponse('Ошибка: данные не найдены в сессии')

    answers = request.session['answers']
    questions = json.loads(request.session['questions'])
    len_answers = len(answers)
    len_questions = len(questions)

    if request.method == "POST":
        if (len_answers < len_questions) and (len_answers + 1 == number):
            save_answer(request)
            len_answers = len(answers)
            print('answers', len_answers)

    if len_answers >= len_questions:
        complete_test(request, answers, questions)
        return redirect(f"{reverse('finish_test')}")
    
    if (number != len_answers + 1):
        redirect_url = f"{reverse('test', args=[len_answers + 1])}"
        return redirect(redirect_url)
    
    if 'lessons' in request.session:
        lessons = json.loads(request.session['lessons'])
    else:
        lessons = [question.name for question in questions]
        lessons.sort()
        request.session['lessons'] = json.dumps(lessons)
        request.session.save()
    
    # it is definitely len_answers < len_questions since we check that condition above
    try:
        question = questions[len_answers]
    except:
        #если сломалось
        question = questions[len_questions-1]
    context = {
        'question': question,
        'number': number,
        'lessons': lessons,
        'all_count':len_questions
        }
    return render(request, 'test.html', context)

def finish_test(request):
    if 'correct_answers' in request.session:
        if 'questions' not in request.session:
            print('error not found')
            return HttpResponse('Ошибка: данные не найдены в сессии')
        questions = json.loads(request.session['questions'])
        len_questions = len(questions)
        correct_answers = request.session['correct_answers']
        return render(request, 'finish_test.html', context={'right_count':correct_answers, 'all_count':len_questions})
    else:
        if 'answers' not in request.session:
            print('error not found')
            return HttpResponse('Ошибка: данные не найдены в сессии')
        answers = request.session['answers']
        len_answers = len(answers)
        number = len_answers + 1
        redirect_url = f"{reverse('test', args=[number])}"
        return redirect(redirect_url)

def save_answer(request):
    answer = request.POST.get('label', None)
    print('answer',answer)
    
    request.session['answers'].append(answer)
    request.session.save()

def complete_test(request, answers, questions):
    len_questions = len(questions)
    print(answers)
    correct_answers = { answers[i] for i in range(len_questions) if questions[i]['name'] == answers[i] }
    print(correct_answers)
        # у вопроса такие поля
        #'id'
        #'name': название лэйбла,
        #'video': путь к файлу
        #'type': тип - show_gesture и guess_gesture
        #Сохранение ответов в базе данных. Вопросы вперемешку
    if 'answers' in request.session:
        del request.session['answers']
    request.session['correct_answers'] = len(correct_answers)
    request.session.save()

    user = request.user
    if user.is_authenticated:
        labels = [question['name'] for question in questions]
        label_objects = Label.objects.filter(name__in = labels)
        user_profile = UserProfile.objects.get(user = user)

        for label in label_objects:
            if user_profile.tests_completed.filter(name = label.name).exists():
                if not (label.name in correct_answers):
                    user_profile.tests_completed.remove(label)
            else:
                if label.name in correct_answers:
                    user_profile.tests_completed.add(label)
        user_profile.save()


class VideoCamera(object):

    def __init__(self, pipeline):
        video = cv2.VideoCapture(0)
        self.video = video
    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, image = self.video.read()
        return image

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()


def gen2(camera,label,user,getResponse):
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]
    frame_counter = 0
    tensors_list = []
    frame_interval = 2
    window_size = 32
    text = ''
    result = ''
    while True:
        frame=camera.get_frame()
        frame_counter+=1
        if frame_counter == frame_interval:
            
            image = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
            image = resize(image, (224, 224))
            image = (image - mean) / std
            image = np.transpose(image, [2, 0, 1])
            tensors_list.append(image)
            if (len(tensors_list)==window_size):
                text = ''
                input_tensor = np.stack(tensors_list[: window_size], axis=1)[None][None]
                input_tensor = input_tensor.astype(np.float32)
                input_tensor = torch.from_numpy(input_tensor)
                with torch.no_grad():
                    outputs = model(input_tensor)[0]
                #tensors_list = np.array(tensors_list)
                #df_video = process_video_arr(tensors_list, 'output_parquet.parquet')
                #X = load_relevant_data_subset_df(df_video)
                #output = prediction_fn(inputs=X)
                #sign = np.argmax(output["outputs"])
                result = str(ORD2SIGN[outputs.argmax().item()])
                #ORD2SIGN[sign]
                text = getText(label, result, user)
                tensors_list.clear()
                #tensors_list = []
            frame_counter = 0
        text_div = np.zeros((200, frame.shape[1], 3), dtype=np.uint8)
        cv2.putText(text_div, text, (300, 80), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 3)
        frame = np.concatenate((frame, text_div), axis=0)
        _, image = cv2.imencode('.jpg', frame)
        cv2.resize(image, (640,480), interpolation = cv2.INTER_AREA)
        yield getResponse(image, result)

def gen3(camera,label,user,getResponse):
    frame_counter = 0
    tensors_list = []
    frame_interval = 3
    window_size = 32
    text = ''
    result = ''
    while True:
        frame=camera.get_frame()
        frame_counter+=1
        if frame_counter == frame_interval:
            tensors_list.append(frame)
            if (len(tensors_list)==window_size):
                text = ''
                tensors_list = np.array(tensors_list)
                df_video = process_video_arr(tensors_list, 'output_parquet.parquet')
                X = load_relevant_data_subset_df(df_video)
                output = prediction_fn(inputs=X)
                sign = np.argmax(output["outputs"])
                result = ORD2SIGN[sign]
                text = getText(label, result, user)
                tensors_list = []
            frame_counter = 0
        text_div = np.zeros((200, frame.shape[1], 3), dtype=np.uint8)
        cv2.putText(text_div, text, (300, 80), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 3)
        frame = np.concatenate((frame, text_div), axis=0)
        _, image = cv2.imencode('.jpg', frame)
        cv2.resize(image, (640,480), interpolation = cv2.INTER_AREA)
        yield getResponse(image, result)

def gen(camera,label,user,getResponse):
    frame_counter = 0
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]
    tensors_list = []
    tensors_list2 = []
    frame_interval = 2
    window_size = 32
    text = ''
    result = ''
    while True:
        frame=camera.get_frame()
        frame_counter+=1
        if frame_counter == frame_interval:
            tensors_list2.append(frame)
            image = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
            image = resize(image, (224, 224))
            image = (image - mean) / std
            image = np.transpose(image, [2, 0, 1])
            tensors_list.append(image)
            if len(tensors_list) == window_size:
                tensors_list2 = np.array(tensors_list2)
                df_video = process_video_arr(tensors_list2, 'output_parquet.parquet')
                X = load_relevant_data_subset_df(df_video)
                output = prediction_fn(inputs=X)
            #print(output)
                #sign = np.argmax(output["outputs"])
                #result = classes[sign]
                #print('My network', result)
                input_tensor = np.stack(tensors_list[: window_size], axis=1)[None][None]
                input_tensor = input_tensor.astype(np.float32)
                input_tensor = torch.from_numpy(input_tensor)
                with torch.no_grad():
                    outputs = model(input_tensor)[0]
                #print(outputs)
                #gloss = str(classes[outputs.argmax().item()])
                #print('Swin',gloss)
                outputs_my = output["outputs"]
                outputs_swin = outputs.numpy()
                result_sign = np.argmax(0.5*outputs_my+outputs_swin*0.5)
                result = ORD2SIGN[result_sign]
                #print('common',classes[common])
                text = getText(label, result, user)
                #print('--------------------------')
                tensors_list = []
                tensors_list2 = []
                
            frame_counter = 0
            text_div = np.zeros((200, frame.shape[1], 3), dtype=np.uint8)
            cv2.putText(text_div, text, (300, 80), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 3)
            frame = np.concatenate((frame, text_div), axis=0)
            _, image = cv2.imencode('.jpg', frame)
            cv2.resize(image, (640,480), interpolation = cv2.INTER_AREA)
            yield getResponse(image, result)
def getStringResponse(image, result):
    image_data_base64 = base64.b64encode(image.tobytes()).decode('utf-8')
    result_data = result

    return 'data: {}\n\n'.format(json.dumps({'image': image_data_base64, 'result': result_data}))

def getByteResponse(image, result):
    return (b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + image.tobytes() + b'\r\n\r\n')



def livefe(request,label):
    try:
        cam = VideoCamera(0)
        user = request.user
        return StreamingHttpResponse(gen(cam,label,user,getByteResponse), content_type="multipart/x-mixed-replace;boundary=frame")
    except: 
        pass

def livefe_test(request,label):
    try:
        cam = VideoCamera(0)
        user = request.user
        return StreamingHttpResponse(gen(cam,label,user,getStringResponse), content_type='text/event-stream')
    except:  
        pass
