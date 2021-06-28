import json
from progress.bar import IncrementalBar
import os
import mediapipe as mp
import cv2

images_path = "/home/bvdvlg/PycharmProjects/diplom/asl_alphabet_train/Training Set"
dots_path = "/home/bvdvlg/PycharmProjects/diplom/asl_alphabet_train/asl_alphabet_train_dots"


class Constants:
    MIN_DETECTION_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5
    BUFFER_SIZE = 10

    class Thresholds:
        PROBA_TRESHOLD = 0.3
        CUTOFF_TRESHOLD = 0.8
        UPPER_TRESHOLD = 4
        LOWER_THRESHOLD = 0.9
        DELAY_TRESHOLD = 1.1
        RECOG_THRESHOLD = 0.3


class LandMark:
    def __init__(self, landmark=None, x=None, y=None, z=None):
        if landmark is None:
            self.x, self.y, self.z = x, y, z
            return
        if isinstance(landmark, dict):
            self.x, self.y, self.z = landmark["x"], landmark["y"], landmark["z"]
            return
        else:
            self.x, self.y, self.z = landmark.x, landmark.y, landmark.z

    def tovec(self):
        return [self.x, self.y, self.z]

    def __repr__(self):
        return str(self.__dict__)


class LandMarks:
    def __init__(self, landmarks):
        self.landmarks = [LandMark(land) for land in landmarks]

    def landstovec(self):
        res = list()
        for elem in self.landmarks:
            res += [elem.x, elem.y, elem.z]
        return res

    def __len__(self):
        return len(self.landmarks)

    def __dict__(self):
        return {"landmarks": [landmark.__dict__ for landmark in self.landmarks], "len": len(self)}

    def __repr__(self):
        return str(self.__dict__())


class Image:
    def __init__(self, width=0, height=0, landmarks=None, gesture=None):
        self.width, self.height = width, height
        self.landmarks = LandMarks(landmarks)
        self.gesture = gesture

    @staticmethod
    def from_image(image):
        return Image(image["width"], image["height"], image["landmarks"]["landmarks"], image["gesture"])

    def __dict__(self):
        return {"width": self.width, "height": self.height, "landmarks": self.landmarks.__dict__(),
                "gesture": self.gesture}

    def fwrite(self, filename):
        with open(filename, "w") as file:
            json.dump(self.__dict__(), file)

    @staticmethod
    def fread(filename):
        with open(filename, "r") as file:
            image = json.load(file)
        return Image.from_image(image)

    def __str__(self):
        return str(self.__dict__())

    def get_data(self):
        return self.landmarks.landstovec(), self.gesture


def images_to_landmarks(impath=images_path, dotpath=dots_path):
    hands = mp.solutions.hands.Hands(min_detection_confidence=Constants.MIN_DETECTION_CONFIDENCE,
                                          min_tracking_confidence=Constants.MIN_TRACKING_CONFIDENCE,
                                          max_num_hands=2)

    suffixes = sorted(os.listdir(impath))
    log_file = "log.txt"
    stop_word = "finished"

    if not os.path.exists("log.txt"):
        with open(log_file, "w") as log:
            log.write(suffixes[0])

    with open(log_file, "r") as log:
        last = log.read()

    if last == stop_word:
        with open(log_file, "w") as log:
            log.write(suffixes[0])

    for suffix in suffixes[suffixes.index(last):]:
        bar = IncrementalBar(suffix, max=len(os.listdir("{}/{}".format(impath, suffix))))
        dirs = os.listdir("{}/{}".format(impath, suffix))
        with open(log_file, "w") as log:
            log.write(suffix)
        for file in dirs:
            bar.next()
            image = image_to_landmarks("{}/{}/{}".format(impath, suffix, file), hands)
            if image is not None:
                image[0].gesture = suffix
                image[0].fwrite("{}/{}/{}.json".format(dotpath, suffix, file[:file.index(".")]))
        bar.finish()
    with open(log_file, "w") as log:
        log.write(stop_word)


def image_to_landmarks(file_list, hands):
    if not isinstance(file_list, list):
        file_list = [file_list, ]
    res = list()
    for idx, file in enumerate(file_list):
        image = cv2.flip(cv2.imread(file), 1)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            continue
        hand_landmarks = results.multi_hand_landmarks[0]
        img = Image(image.shape[0], image.shape[1], hand_landmarks.landmark, None)
        res.append(img)
    if len(res) == 0:
        return None
    return res
