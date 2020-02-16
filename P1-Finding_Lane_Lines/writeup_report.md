# **Finding Lane Lines on the Road** 

## Writeup Report

### This file descibes in short lines what have been accomplished in this project.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflection on a work in a written report


Challange: 
![alt text](https://github.com/krunicm/nanodegree_self-driving/blob/master/P1-Finding_Lane_Lines/test_images/challange2.jpg?raw=true "Challange")

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps:

First, Image is converted into the gray scale:

[//]: # (Image References)

Gray: 
![alt text](https://github.com/krunicm/nanodegree_self-driving/blob/master/P1-Finding_Lane_Lines/processing_images/gray.png?raw=true "Gray")

Second, Gaussian smooting is performed.

Gaussian smooting: 
![alt text](https://github.com/krunicm/nanodegree_self-driving/blob/master/P1-Finding_Lane_Lines/processing_images/blue_gray.png?raw=true "Gray")

Third, Canny Edge Detection is applied.

Canny Edge Detection:
![alt text](https://github.com/krunicm/nanodegree_self-driving/blob/master/P1-Finding_Lane_Lines/processing_images/edges.png?raw=true "Canny")

Fourth, Cropping the Region of Interest is performed.

Cropping the Region of Interest:
![alt text](https://github.com/krunicm/nanodegree_self-driving/blob/master/P1-Finding_Lane_Lines/processing_images/roi.png?raw=true "ROI")

Fifth, Lines are annotated on the road.

Annotating lines on the road:
![alt text](https://github.com/krunicm/nanodegree_self-driving/blob/master/P1-Finding_Lane_Lines/processing_images/lines_annotated.png?raw=true "Lines")

There was two most challaning parts of the project:
1. Tunning the parametars
2. Modifying the "draw_lines()" function

Tunning parametars has been performed by the tool developed for that purpose, which can be found in the "tools" folder. This tool represents just enhanced version of the tool created by Maunesh Ahir:
https://github.com/maunesh/opencv-gui-helper-tool

Tool for parameters tuning:
![alt text](https://github.com/krunicm/nanodegree_self-driving/blob/master/P1-Finding_Lane_Lines/processing_images/tool.png?raw=true "Tool")

Improvement of the "draw_line()" function has been done mostly by reading instructions from this tutorial:
https://medium.com/@mrhwick/simple-lane-detection-with-opencv-bfeb6ae54ec0

Finaly, pipline processing has been performed over test videos, and this is the output:

[![Annotated "solidWhiteRight.mp4"](http://img.youtube.com/vi/IcTcruQdQCA/0.jpg)](https://youtu.be/IcTcruQdQCA)

[![Annotated "solidYellowLeft.mp4"](http://img.youtube.com/vi/WtfqIXzRVHk/0.jpg)](https://youtu.be/WtfqIXzRVHk)

[![Annotated "challenge.mp4"](http://img.youtube.com/vi/5z_z8MCQSfI/0.jpg)](https://youtu.be/5z_z8MCQSfI)



### 2. Identify potential shortcomings with your current pipeline


...


### 3. Suggest possible improvements to your pipeline

...
