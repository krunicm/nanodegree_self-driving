"""
How to run:
python find_edges.py <image path>
"""

import argparse
import cv2
import os
import helper_functions as help_fun
import numpy as np
import matplotlib.image as mpimg

from guiutils import EdgeFinder


def main():
    # parser = argparse.ArgumentParser(description='Visualizes the line for hough transform.')
    # parser.add_argument('filename')

    # args = parser.parse_args()

#reading in an image
# challange1.jpg
# challange2.jpg
# challange3.jpg
# solidWhiteCurve.jpg
# solidWhiteRight.jpg
# solidYellowCurve.jpg
# solidYellowCurve2.jpg
# solidYellowLeft.jpg
# whiteCarLaneSwitch.jpg

    filename = "test_images/challange2.jpg"
    img = mpimg.imread(filename)
    edge_finder = EdgeFinder(img, filter_size = 5, 
                                    threshold1 = 88, 
                                    threshold2 = 133, 
                                    rho = 7, 
                                    theta_coef=1, 
                                    threshold = 97,
                                    min_line_length = 197,
                                    max_line_gap = 1000)

    print("Edge parameters:")
    print("GaussianBlur Filter Size: %f" % edge_finder.filterSize())
    print("Threshold1: %f" % edge_finder.threshold1())
    print("Threshold2: %f" % edge_finder.threshold2())
    print("Rho: %f" % edge_finder.rho())
    print("Theta coef: %f" % edge_finder.theta_coef())
    print("Min line lenght: %f" % edge_finder.min_line_lenght())
    print("Max line gap: %f" % edge_finder.max_line_gap())

    (head, tail) = os.path.split(filename)
    (root, ext) = os.path.splitext(tail)
 
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
