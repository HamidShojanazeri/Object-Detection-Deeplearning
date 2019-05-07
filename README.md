# Documentation

This project includes functions to perfrom the task of detecting a car in a specific parking lot. The following are explanations about the structure of the project and its usage.



## Structure

This project includes a main module with the name of cars.py. This module provides three different functions: detect, compare and analyse. 

This project also includes a docker file along with the requirement.txt used for running through docker.

It has a seperate folder (yolo-coco) that keeps the yolov3 config file along with the coco dataset labels. The cars.py downloads the Yolov3 weights and save it in this folder as well.

There is a cache folder that will be produced during the runtime which includes the verkada sample video index file and the first frame of each of the sample videos which passed to the functions for processing.

## Usage

Three functions provided in the cars.py can be used as follows:

- detect 1538076003

This function accepts a timestamp and run the yolov3 model to check if a car is parked in the defined parking spot.

- Compare 1538076003 1538076083

This function accepts two timestamps and compare if two images are the same in the defined spot. If they are not the same then it calls detect function to check if the parking sopt is occupied with a new car or it is free. 

- analyze 1538076003 1538076083

This function accepts two timestamps and check the defined parking sopt between the two timestamps, if it detects a car parked in the spot then it returns the time that the car has been parked there until it left. Then if a new car come to the sopt again detects it and calculate its parking time until it reaches to the second timestamp passed to the funciton. 

The output show that at which frame (timestamp) a car has been detected and that car has been parked until what timestamp along with the time between them in minutes. It also shows the time stamps that the parking was not occupied.

