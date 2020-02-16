import cv2
import helper_functions as help_fun
import numpy as np

class EdgeFinder:
    def __init__(self, image, 
                filter_size=1, 
                threshold1=0, 
                threshold2=0, 
                rho=5, 
                theta_coef=1, 
                threshold=84, 
                min_line_length = 120, 
                max_line_gap=1000):

        self.image = image
        self._filter_size = filter_size
        self._threshold1 = threshold1
        self._threshold2 = threshold2
        self._rho = rho
        self._theta_coef = theta_coef
        self._threshold = threshold
        self._min_line_length = min_line_length
        self._max_line_gap = max_line_gap

        def onchangeThreshold1(pos):
            self._threshold1 = pos
            self._render()

        def onchangeThreshold2(pos):
            self._threshold2 = pos
            self._render()

        def onchangeFilterSize(pos):
            self._filter_size = pos
            self._filter_size += (self._filter_size + 1) % 2
            self._render()

        def onchangeRho(pos):
            self._rho = pos
            self._render()

        def onchangeThetaCoef(pos):
            self._theta_coef = pos
            self._render()        

        def onchangeThreshold(pos):
            self._threshold = pos
            self._render()

        def onchangeMinLineLenght(pos):
            self._min_line_length = pos
            self._render()

        def onchangeMaxLineGap(pos):
            self._max_line_gap = pos
            self._render()

        cv2.namedWindow('edges')

        cv2.createTrackbar('threshold1', 'edges', self._threshold1, 255, onchangeThreshold1)
        cv2.createTrackbar('threshold2', 'edges', self._threshold2, 255, onchangeThreshold2)
        cv2.createTrackbar('filter_size', 'edges', self._filter_size, 20, onchangeFilterSize)
        cv2.createTrackbar('rho', 'edges', self._rho, 360, onchangeRho)
        cv2.createTrackbar('theta_coef', 'edges', self._theta_coef, 360, onchangeThetaCoef)
        cv2.createTrackbar('threshold', 'edges', self._threshold, 255, onchangeThreshold)
        cv2.createTrackbar('min_line_length', 'edges', self._min_line_length, 2000, onchangeMinLineLenght)
        cv2.createTrackbar('max_line_gap', 'edges', self._max_line_gap, 2000, onchangeMaxLineGap)

        self._render()

        print("Adjust the parameters as desired.  Hit any key to close.")

        cv2.waitKey(0)

        cv2.destroyWindow('edges')
        cv2.destroyWindow('smoothed')

    def threshold1(self):
        return self._threshold1

    def threshold2(self):
        return self._threshold2

    def filterSize(self):
        return self._filter_size

    def rho(self):
        return self._rho

    def theta_coef(self):
        return self._theta_coef

    def threshold(self):
        return self._threshold

    def min_line_lenght(self):
        return self._min_line_length

    def max_line_gap(self):
        return self._max_line_gap

    def edgeImage(self):
        return self._edge_img

    def smoothedImage(self):
        return self._smoothed_img

    def _render(self):
        weighted_imgage = help_fun.process_image(self.image,
                                                self._filter_size, 
                                                self._threshold1, 
                                                self._threshold2, 
                                                self._rho, 
                                                self._theta_coef, 
                                                self._threshold, 
                                                self._min_line_length, 
                                                self._max_line_gap)

        # cv2.imshow('lines', lines)
        # cv2.imshow('smoothed', self._smoothed_img)
        # cv2.imshow('edges', self._edge_img)
        cv2.imshow('edges', weighted_imgage)
