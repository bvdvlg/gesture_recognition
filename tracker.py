import time
from time import time
from image_container import LandMarks, LandMark
import numpy as np
from tube import Tube
import numpy.fft as fourier
buffer_size = 10
upper_threshhold = 6
lower_threshhold = 0.8
'''
gesture can be recognized = 1
gesture was read = 0
speed was detected = -1
'''


start_time = time()


class Tracker:
    def __init__(self):
        self.speed_hist = Tube.get_tube([0]*buffer_size)
        self.last_time = time()
        self.counted_idx = [4, 3, 6, 8, 10, 12, 14, 16, 18, 20]
        self.flag = 0
        self.speed = list()
        self.weights = np.array([2, 2, 0.5]+[1]*27)
        self._num_counted = len(self.counted_idx)
        self.last_coordinates = np.array(self._num_counted*[0])#np.array(self._num_counted*3*[0])

    def update(self, landmarks):
        speed, vec_of_coords, new_time = self.get_speed(landmarks)
        speed = max(speed)
        if self.flag == 0 and speed > upper_threshhold:
            self.flag = -1
        if self.flag == -1 and speed < lower_threshhold:
            self.flag = 1
        self.speed_hist = self.speed_hist.pushed(speed)
        self.speed_hist.pop()
        self.last_time = new_time
        self.last_coordinates = vec_of_coords
        return self

    def get_speed(self, landmarks):
        new_time = time()
        dist = ((landmarks[5].x - landmarks[0].x) ** 2 + (landmarks[5].y - landmarks[0].y) ** 2 + (
                    landmarks[5].z - landmarks[0].z) ** 2) ** (1 / 2)
        used_lands = LandMarks([landmarks[i] for i in self.counted_idx]).landstovec()
        zero_point = np.mean([np.mean(used_lands[::3]), np.mean(used_lands[1::3]), np.mean(used_lands[2::3])])
        used_lands = np.array(used_lands) * self.weights
        distances = [np.linalg.norm(used_lands[3*i:3*i+3]-zero_point)/dist for i in range(self._num_counted)]
        speed = (np.array(distances)-np.array(self.last_coordinates)) / (new_time - self.last_time)
        return np.abs(speed), distances, new_time

    def ft(self):
        res = np.fft.fft(self.speed_hist.tolist())
        return res