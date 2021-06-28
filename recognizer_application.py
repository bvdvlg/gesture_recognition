import tkinter
import cv2
import mediapipe as mp
from image_container import Constants
import json
from models import XGBModel
import gui_module
from tracker import Tracker
from text import Text


class App:
    def __init__(self, window, window_title, video_source=0):
        self.video_source = video_source
        self.model = XGBModel.get_model('xgb_model.pkl')
        self.text = Text()
        with open('labels.json', 'r') as labels_file:
            labels = json.load(labels_file)
        self.text.set_alphabets(labels)
        self.vid = VideoCapture(self.video_source)
        self.tracker = Tracker()
        self.photo = None
        self.GUI = gui_module.GUI(self, window, window_title)
        self.text.text += self.text.current_word
        del self.vid


class VideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = mp.solutions.hands.Hands(min_detection_confidence=Constants.MIN_DETECTION_CONFIDENCE,
                                              min_tracking_confidence=Constants.MIN_TRACKING_CONFIDENCE,
                                              max_num_hands=2)

        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self, draw_marks=1):
        if not self.vid.isOpened():
            raise Exception("camera is not available")

        ret, image = self.vid.read()

        if not ret:
            return ret, None, None

        # Read the image
        if draw_marks:
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            # Get hand landmarks
            image.flags.writeable = False
            results = self.hands.process(image)

            if not results.multi_hand_landmarks:
                return ret, image, None

            return ret, image, results.multi_hand_landmarks
        else:
            return ret, image, None

    def draw_landmarks(self, image, landmarks):
        self.mp_drawing.draw_landmarks(image, landmarks, mp.solutions.hands.HAND_CONNECTIONS)

    def draw_sym(self, image, sym):
        cv2.putText(image, str(sym),
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, color=(255, 0, 0))

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


# Create a window and pass it to the Application object
App(tkinter.Tk(), "Tkinter and OpenCV")
