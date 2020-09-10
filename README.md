# Last-Mile Autonomous Delivery Robot Computer Vision System

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Our full report: https://github.com/quan-possible/lmad-protopaja/blob/master/Final.pdf

This repository contains the source code of our project - an attempt at building a computer vision system for self-driving delivery robots. The goal is to create a system that enables such machines to navigate the real world safely and efficiently, solving a task that was considered almost impossible until recently. Within the system, there are 3 fundamental components:

## Path Segmentation
In this module, a **UNet with attention gate** is used to perform **semantic segmentation** and **drivable area segmentation**, which help build the understanding of the vehicle about its surrounding environment. Two models presented here were trained on [CityScapes](https://www.cityscapes-dataset.com/) dataset and [Berkeley Deep Drive](https://bdd-data.berkeley.edu/) dataset.

![](https://raw.githubusercontent.com/quan-possible/lmad-protopaja/master/images/rsz_screenshot_from_2020-09-04_11-59-43.png)
## Object Detection
To ensure the safety of the pedestrians as well as the vehicle itself, **detecting and avoiding obstacles** are the most critical tasks for autonomous vehicles. Our module use [Intel Realsense Depth Camera](https://www.intelrealsense.com/stereo-depth/?utm_source=intelcom_website&utm_medium=button&utm_campaign=day-to-day&utm_content=D400_learn-more_button) and several computer vision algorithms to detect obstacles. 
![](https://raw.githubusercontent.com/quan-possible/lmad-protopaja/master/images/rsz_screenshot_from_2020-09-04_12-04-34.png)
## Path Planning
With the perception of the two previous modules, **navigating through the vehicle's local environment** is our final goal. The path planning module employs [Weighted A*](http://theory.stanford.edu/~amitp/GameProgramming/AStarComparison.html) algorithm to find the shortest path from the vehicle current location to a front target. 
![](https://raw.githubusercontent.com/quan-possible/lmad-protopaja/master/images/rsz_screenshot_from_2020-09-04_12-15-08.png)

The work was done in collaboration with Aalto University and Futurice Oy.
