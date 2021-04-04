import time
from time import time
from image_container import LandMarks, LandMark
import numpy as np
from tube import Tube
import numpy.fft as fourier
from image_container import Constants
import json
'''
gesture can be recognized = 1
gesture was read = 0
speed was detected = -1
'''


start_time = time()


class Tracker:
    def __init__(self):
        self.speed_hist = Tube.get_tube([0] * Constants.BUFFER_SIZE)
        self.last_time = time()
        self.counted_idx = [4, 3, 6, 8, 10, 12, 14, 16, 18, 20]
        self.flag = 0
        self.weights = np.array([1.5, 1.5, 1.5]+[1]*27)
        self._num_counted = len(self.counted_idx)
        self.delayer_flag = 0
        self.counter = 0
        self.speed_array = list()
        self.update_delayed_flag()
        self.__stopped_coords = list()
        self.__max_distances = [0]*self._num_counted
        self.last_coordinates = np.array(self._num_counted*[0])
        self.__gesture_delay_counter = 0

    def update(self, landmarks):
        speed_arr, vec_of_coords, new_time = self.get_speed(landmarks)
        speed = max(np.abs(speed_arr))

        if self.delayer_flag != 0:
            self.__stopped_coords = vec_of_coords
            self.__max_distances = [0] * self._num_counted
            self.__gesture_delay_counter = 0

        if self.flag == 0 and speed > Constants.Thresholds.UPPER_TRESHOLD and self.delayer_flag == 0:
            self.flag = -1

        if self.flag == -1:
            diff = (np.array(vec_of_coords)-np.array(self.__stopped_coords))**2
            self.__max_distances = [max(a, b) for a, b in zip(self.__max_distances, diff)]

        if self.flag == -1 and speed < Constants.Thresholds.LOWER_THRESHOLD:
            self.__gesture_delay_counter += 1
            if self.__gesture_delay_counter >= Constants.Thresholds.DELAY_TRESHOLD:
                if max(self.__max_distances)**(1/2) > Constants.Thresholds.RECOG_THRESHOLD:
                    self.flag = 1
                self.__stopped_coords = vec_of_coords
                self.__max_distances = [0]*self._num_counted
                self.__gesture_delay_counter = 0

        self.speed_hist = self.speed_hist.pushed(speed_arr)
        self.speed_hist.pop()
        self.last_time = new_time
        self.last_coordinates = vec_of_coords
        return self

    def get_speed(self, landmarks):
        new_time = time()
        dist = ((landmarks[5].x - landmarks[0].x) ** 2 + (landmarks[5].y - landmarks[0].y) ** 2 + (
                    landmarks[5].z - landmarks[0].z) ** 2) ** (1 / 2)
        used_lands = LandMarks([landmarks[i] for i in self.counted_idx]).landstovec()
        zero_point = [landmarks[0].x, landmarks[0].y, landmarks[0].z]
        used_lands = np.array(used_lands) * self.weights
        distances = [np.linalg.norm(used_lands[3*i:3*i+3]-zero_point)/dist for i in range(self._num_counted)]
        speed = (np.array(distances)-np.array(self.last_coordinates)) / (new_time - self.last_time)
        return speed, distances, new_time

    def write(self, filename):
        with open(filename, "w") as file:
            json.dump({"data": self.speed_array}, file)

    def load(self, filename):
        with open(filename, "r") as file:
            self.speed_array = json.load(file)

    def update_delayed_flag(self):
        self.delayer_flag = 3*Constants.BUFFER_SIZE
        self.flag = 0

    def ft(self):
        res = np.fft.fft(self.speed_hist.tolist())
        return res