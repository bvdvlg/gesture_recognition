import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
import cv2
import mediapipe as mp
import numpy as np
import data
import image_container
from image_container import Constants
import tkinter.ttk as ttk
import time
import os
from models import XGBModel, FCModel, KNN, RandomForestModel
import pandas as pd
from tracker import Tracker
from text import Text


class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.model = XGBModel.get_model('xgb_model.pkl')
        self.text = Text()
        labels = {el.lower(): num for num, el in enumerate(sorted(os.listdir(image_container.dots_path)))}
        self.text.set_alphabets(labels)
        self.tracker = Tracker()
        self.photo = None

        # open video source (by default this will try to open the computer webcam)
        self.vid = VideoCapture(self.video_source)
        self.window.geometry('{}x{}'.format(int(self.vid.width+20), int(self.vid.height+110)))
        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(self.window, width=self.vid.width, height=self.vid.height)
        self.canvas.place(x=0, y=50)

        label = tkinter.Label(self.window, text="upper_threshold")
        label.place(x=self.vid.width//2, y=10)
        self.upper_theshold_label = ttk.Label(self.window, text=str(Constants.Thresholds.upper_threshhold))
        self.upper_theshold_label.place(x=self.vid.width-20, y=30)

        self.upper_scale = ttk.Scale(self.window, from_=1, to=30, command=self.upper_scaler, length=self.vid.width-30)
        self.upper_scale.place(x=10, y=30)
        self.upper_scale.set(Constants.Thresholds.upper_threshhold)

        label = tkinter.Label(self.window, text="lower_threshold")
        label.place(x=self.vid.width // 2, y=50)
        self.lower_theshold_label = ttk.Label(self.window, text=str(Constants.Thresholds.lower_threshhold))
        self.lower_theshold_label.place(x=self.vid.width-20, y=70)

        self.lower_scale = ttk.Scale(self.window, from_=0.1, to=10, command=self.lower_scaler, length=self.vid.width-30)
        self.lower_scale.place(x=10, y=70)
        self.lower_scale.set(Constants.Thresholds.lower_threshhold)

        self.current_word_label = tkinter.Label(self.window, text="current word:")
        self.common_text_label = tkinter.Label(self.window, text="text:")
        self.current_word_label.place(x=10, y=90)
        self.common_text_label.place(x=10, y=110)

        self.current_word_label_info = tkinter.Label(self.window, text=self.text.current_word)
        self.common_text_label_info = tkinter.Label(self.window, text=self.text.text)
        self.current_word_label_info.place(x=100, y=90)
        self.common_text_label_info.place(x=45, y=110)

        self.button = tkinter.Checkbutton(self.window, text="Do you want to use t9?", command=self.Button_t9)
        self.button.place(x=50, y=550)
        self.t9label = tkinter.Label(self.window, text="Yes" if self.text.use_t9 else "No")
        self.button.select()
        self.t9label.place(x=20, y=550)

        self.clear_button = tkinter.Button(self.window, text="Clear", command=self.clear_button_allert)
        self.clear_button.place(x=400, y=550)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 5
        self.update()

        self.window.mainloop()
        self.text.text += self.text.current_word

    def clear_button_allert(self):
        self.text.text = ""
        self.text.current_word = ""
        self.common_text_label_info["text"] = self.text.text + self.text.current_word
        self.current_word_label_info["text"] = self.text.current_word

    def Button_t9(self):
        self.text.use_t9 = not self.text.use_t9
        self.t9label["text"] = "Yes" if self.text.use_t9 else "No"

    def upper_scaler(self, val):
        Constants.Thresholds.upper_threshhold = float(val)
        self.upper_theshold_label["text"] = "%.2f" % Constants.Thresholds.upper_threshhold

    def lower_scaler(self, val):
        Constants.Thresholds.lower_threshhold = float(val)
        self.lower_theshold_label["text"] = "%.2f" % Constants.Thresholds.lower_threshhold

    def update(self):
        # Get a frame from the video source
        ret, frame, marks = self.vid.get_frame()

        if marks is not None:
            marks = marks[0]
            self.tracker.update(marks.landmark)
            self.vid.draw_landmarks(frame, marks)
            if self.tracker.flag == 1 and self.tracker.delayer_flag == 0:
                out, probas = self.model.lands2sym(marks.landmark)
                sym = self.text.get_sym(out)

                if probas[0][int(out)] > Constants.Thresholds.proba_threshold:
                    self.text.append(sym)
                    self.tracker.flag = 0

                self.common_text_label_info["text"] = self.text.text+self.text.current_word
                self.current_word_label_info["text"] = self.text.current_word

            if self.tracker.delayer_flag > 0:
                self.tracker.delayer_flag -= 1
        else:
            self.tracker.update_delayed_flag()
            self.text.current_word = ""
            self.current_word_label_info["text"] = self.text.current_word

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(10, 100, image=self.photo, anchor=tkinter.NW)

        self.window.after(self.delay, self.update)


class VideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = mp.solutions.hands.Hands(min_detection_confidence=Constants.min_detection_confidence,
                                              min_tracking_confidence=Constants.min_tracking_confidence,
                                              max_num_hands=1)

        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if not self.vid.isOpened():
            raise Exception("camera is not available")

        ret, image = self.vid.read()

        if not ret:
            return ret, None, None

        # Read the image
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # Get hand landmarks
        image.flags.writeable = False
        results = self.hands.process(image)

        if not results.multi_hand_landmarks:
            return ret, image, None

        return ret, image, results.multi_hand_landmarks

    def draw_landmarks(self, image, landmarks):
        self.mp_drawing.draw_landmarks(image, landmarks, mp.solutions.hands.HAND_CONNECTIONS)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


# Create a window and pass it to the Application object
App(tkinter.Tk(), "Tkinter and OpenCV")
