# Last Mile Autonomous Delivery 

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

This repository demonstrates result from our attempt to build self-driving vehicle. The project is in colaboration with Aalto Univeristy and Futurice, where the goal is to explore different approaches to build autonomous vehicle or robot. Further devlopment will push this work to its fullest potential. In this work, 3 submodules with different functionalities are developed to guide the vehicle. Here is the link to our [final report](https://github.com/quan-possible/lmad-protopaja/blob/master/Final.pdf).

## Path Segmentation
In this module, a **UNet with attention gate** is used to perform **semantic segmentation** and **drivable area segmentation**, which help build the understanding of the vehicle about its surrouding environment. Two models presented here was trained on [CityScapes](https://www.cityscapes-dataset.com/) dataset and [Berkeley Deep Drive](https://bdd-data.berkeley.edu/) dataset.
![](https://raw.githubusercontent.com/quan-possible/lmad-protopaja/master/images/rsz_screenshot_from_2020-09-04_11-59-43.png)
## Object Detection
In order to ensure the safety for the pedestrians as well as the vehicle itself, **detecting and avoiding obstacles** are most critical tasks for autonomous vehicles. Our module use [Intel Realsense Depth Camera](https://www.intelrealsense.com/stereo-depth/?utm_source=intelcom_website&utm_medium=button&utm_campaign=day-to-day&utm_content=D400_learn-more_button) and several computer vision algorithms to detect obstacles. 
![](https://raw.githubusercontent.com/quan-possible/lmad-protopaja/master/images/rsz_1screenshot_from_2020-09-04_12-04-34.png)
## Path Following
With the perception from the two previous modules, **navigating through the vehicle's local environment** is our final goal. The path planning module employs [Weighted A*](http://theory.stanford.edu/~amitp/GameProgramming/AStarComparison.html) algorithm to find the shortest path from the vehicle current location to a front target. 
![](https://raw.githubusercontent.com/quan-possible/lmad-protopaja/master/images/rsz_1screenshot_from_2020-09-04_12-15-08.png)
