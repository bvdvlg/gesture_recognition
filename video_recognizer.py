import cv2
import mediapipe as mp
import numpy as np
import data
import image_container
import time
import os
from models import XGBModel, FCModel, KNN, RandomForestModel
import pandas as pd
from tracker import Tracker
from text import Text

proba_threshold = 0.1


def read_image(cap):
    success, image = cap.read()

    # Read the image
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # Get hand landmarks
    image.flags.writeable = False
    results = hands.process(image)

    # BRG image to RGB
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results, success


def make_prediction(model, lands):
    marks = image_container.LandMarks(lands).landstovec()
    marks = data.xgb_data_transformer(marks)
    marks = pd.DataFrame(marks).transpose()
    probas = model.model.predict_proba(marks)
    out = np.argmax(probas[0])
    return int(out), probas


# Hand tracking model defining
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.3, min_tracking_confidence=0.7, max_num_hands=1)

# Text, recognizer and dictionaries defining
text = Text()
xgb = XGBModel()
xgb.load('xgb_model.pkl')
labels = {el.lower(): num for num, el in enumerate(sorted(os.listdir(image_container.dots_path)))}
text.set_alphabets(labels)
track = Tracker()
counter = 0

# Start working
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)

while cap.isOpened():
    image, results, success = read_image(cap)
    if results.multi_hand_landmarks:
        track.update(results.multi_hand_landmarks[0].landmark)
        if counter < buffer_size:
            counter += 1
        else:
            break

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

track.flag = 0
start = time.time()
while cap.isOpened():
    image, results, success = read_image(cap)
    if not success:
        print("Ignoring empty camera frame.")
        continue

    if results.multi_hand_landmarks:
        track.update(results.multi_hand_landmarks[0].landmark)
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if track.flag == 1:
                out, probas = make_prediction(xgb, hand_landmarks.landmark)
                sym = text.get_sym(out)

                if probas[0][int(out)] > proba_threshold:
                    text.append(sym)
                    track.flag = 0

                image = cv2.putText(image, "{} {}".format(sym,
                                                          probas[0][int(out)]), (20, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (235, 22, 255), 4)
    image = cv2.putText(image, "fps: %.2f" % fps, (20, image.shape[1] - 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (235, 22, 255), 4)
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

    fps = 1 / (time.time() - start)
    start = time.time()
hands.close()
cap.release()
print(text.text)
