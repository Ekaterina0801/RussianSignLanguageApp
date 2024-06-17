import cv2
import cv2
import mediapipe as mp
import pandas as pd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os
import torch
from django.db.models import Count
from sign.models import *
import gdown
window_size = 32
#output_names = [output.name for output in session.get_outputs()]

threshold = 0.5
frame_interval = 2
ROWS_PER_FRAME = 543
def resize(im, new_shape=(224, 224)):
    """
    Resize and pad image while preserving aspect ratio.

    Parameters
    ----------
    im : np.ndarray
        Image to be resized.
    new_shape : Tuple[int]
        Size of the new image.

    Returns
    -------
    np.ndarray
        Resized image.
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # add border
    return im


# Load MediaPipe Holistic model
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


# Initialize MediaPipe Holistic model
holistic = mp_holistic.Holistic()

def process_video(video_path, output_path):
    # Initialize video reader
    video_reader = cv2.VideoCapture(video_path)

    frames = []
    frame_index = 0

    while video_reader.isOpened():
        # Read a frame from the video
        success, frame = video_reader.read()
        if not success:
            break

        # Convert the frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run holistic estimation on the frame
        results = holistic.process(frame_rgb)
        # Extract face landmarks
        if results.face_landmarks:
            for i, landmark in enumerate(results.face_landmarks.landmark):
                frames.append([frame_index, f'{frame_index}_face_{i}', 'face', i, landmark.x, landmark.y, landmark.z])
        else:
            for i in range (468):
                frames.append([frame_index, f'{frame_index}_face_{i}', 'face', i, 0, 0, 0])
        # Extract hand landmarks
        if results.left_hand_landmarks:
            for i, landmark in enumerate(results.left_hand_landmarks.landmark):
                frames.append([frame_index, f'{frame_index}_left_hand_{i}', 'left_hand', i, landmark.x, landmark.y, landmark.z])
        else:
            for i in range (21):
                frames.append([frame_index, f'{frame_index}_left_hand_{i}', 'left_hand', i, 0, 0, 0])
        
        if results.right_hand_landmarks:
            for i, landmark in enumerate(results.right_hand_landmarks.landmark):
                frames.append([frame_index, f'{frame_index}_right_hand_{i}', 'right_hand', i, landmark.x, landmark.y, landmark.z])
        else:
            for i in range (21):
                frames.append([frame_index, f'{frame_index}_right_hand_{i}', 'right_hand', i, 0, 0, 0])
        
        if results.pose_landmarks:
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                frames.append([frame_index, f'{frame_index}_pose_{i}', 'pose', i, landmark.x, landmark.y, landmark.z])
        else:
            for i in range (33):
                frames.append([frame_index, f'{frame_index}_pose_{i}', 'pose', i, 0, 0, 0])
        frame_index += 1

    # Release resources
    video_reader.release()

    # Create a dataframe from the extracted landmarks
    df = pd.DataFrame(frames, columns=['frame', 'row_id', 'type', 'landmark_index', 'x', 'y', 'z'])

    # Save the dataframe to a parquet file
    df.to_parquet(output_path)

interpreter = tf.lite.Interpreter("../SignLanguageApp/sign/model_final_1001_depth_best.tflite")
#interpreter = tf.lite.Interpreter("../SignLanguageApp/sign/model_common.tflite")
found_signatures = list(interpreter.get_signature_list().keys())
prediction_fn = interpreter.get_signature_runner("serving_default")
file_id1 = "1DyMrIdBJg9UAj_sWYzS2ZvXDBxBl9SBJ"  
if not os.path.isfile(os.getcwd()+'/sign/end2end-6.pt'):
    gdown.download(f"https://drive.google.com/uc?id={file_id1}", os.getcwd()+'/sign/end2end-6.pt')
model = torch.jit.load(os.getcwd()+"/sign/end2end-6.pt")
model.eval()

def process_video_arr(framess, output_path):
    # Initialize video reader
    #video_reader = cv2.VideoCapture(video_path)

    frames = []
    frame_index = 0

    for frame in framess:
        # Read a frame from the video
        #success, frame = video_reader.read()
        #if not success:
            #break

        # Convert the frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        # Run holistic estimation on the frame
        results = holistic.process(frame_rgb)
        # Extract face landmarks
        if results.face_landmarks:
            for i, landmark in enumerate(results.face_landmarks.landmark):
                frames.append([frame_index, f'{frame_index}_face_{i}', 'face', i, landmark.x, landmark.y, landmark.z])
        else:
            for i in range (468):
                frames.append([frame_index, f'{frame_index}_face_{i}', 'face', i, 0, 0, 0])
        # Extract hand landmarks
        if results.left_hand_landmarks:
            for i, landmark in enumerate(results.left_hand_landmarks.landmark):
                frames.append([frame_index, f'{frame_index}_left_hand_{i}', 'left_hand', i, landmark.x, landmark.y, landmark.z])
        else:
            for i in range (21):
                frames.append([frame_index, f'{frame_index}_left_hand_{i}', 'left_hand', i, 0, 0, 0])
        
        if results.right_hand_landmarks:
            for i, landmark in enumerate(results.right_hand_landmarks.landmark):
                frames.append([frame_index, f'{frame_index}_right_hand_{i}', 'right_hand', i, landmark.x, landmark.y, landmark.z])
        else:
            for i in range (21):
                frames.append([frame_index, f'{frame_index}_right_hand_{i}', 'right_hand', i, 0, 0, 0])
        
        if results.pose_landmarks:
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                frames.append([frame_index, f'{frame_index}_pose_{i}', 'pose', i, landmark.x, landmark.y, landmark.z])
        else:
            for i in range (33):
                frames.append([frame_index, f'{frame_index}_pose_{i}', 'pose', i, 0, 0, 0])
        frame_index += 1

    # Release resources
    #video_reader.release()

    # Create a dataframe from the extracted landmarks
    df = pd.DataFrame(frames, columns=['frame', 'row_id', 'type', 'landmark_index', 'x', 'y', 'z'])

    # Save the dataframe to a parquet file
    #df.to_parquet(output_path)

    return df

def load_relevant_data_subset_df(df):
    data_columns = ['x', 'y', 'z']
    #data = pd.read_parquet(pq_path, columns=data_columns)
    data = df[data_columns]
    data.loc[data.x.isnull(), ('x')] = 0
    data.loc[data.y.isnull(), ('y')] = 0
    data.loc[data.z.isnull(), ('z')] = 0
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)



def getLabelsByCategory(category: str):
    category_object = Category.objects.get(name=category)
    labels = Label.objects.filter(category=category_object).annotate(lessons_cnt=Count('lesson')).filter(lessons_cnt__gt=0)
    return labels

def getCategorizedLabels(user_profile: UserProfile):
    labels = Label.objects.annotate(lessons_cnt=Count('lesson')).filter(lessons_cnt__gt=0)
    categories = Category.objects.all()
    labels_by_category = {}
    labels_learned_by_category = {}
    tests_completed_by_category = {}
    for category in categories:
        labels_by_category[category.name] = []
        labels_learned_by_category[category.name] = []
        tests_completed_by_category[category.name] = []
    for label in labels:
        labels_by_category[label.category.name].append(label)
    if user_profile != None:
        for label in user_profile.labels_learned.all():
            labels_learned_by_category[label.category.name].append(label)
        for test in user_profile.tests_completed.all():
            tests_completed_by_category[test.category.name].append(test)
    categorized_labels = []
    for category in categories:
        dict = {}
        dict["category_name"] = category.name
        dict["category"] = category
        dict["labels"] = labels_by_category[category.name]
        dict["labels_learned"] = labels_learned_by_category[category.name]
        dict["tests_completed"] = tests_completed_by_category[category.name]
        categorized_labels.append(dict)
    return categorized_labels

def getText(label, value, user):
    text = ''
    if (label!='none'):
        if (value==label):
            text = 'Верно!'
            if user.is_authenticated:
                user_profile = UserProfile.objects.get(user = user)
                if not (user_profile.labels_learned.filter(name = label).exists()):
                    label_object = Label.objects.get(name = label)
                    user_profile.labels_learned.add(label_object)
                    user_profile.save()
        else:
            text = f'Нет, это {value}. Попробуйте еще'
    else:
        text = value
    return text