# USAGE
# python3 cars.py detect timestamp1
# python3 cars.py compare timestamp1 timestamp2
# python3 cars.py analyze timestamp1 timestamp2
# python3 cars.py analyze 1538076003 1538078234 --output result.json

# import the necessary packages
from src import cache
import numpy as np
import argparse
import time
import cv2
import os
import sys
import json
from skimage.measure import compare_ssim as ssim


def get_car_boxes(boxes, classIDs):
    car_boxes = []
    for i, box in enumerate(boxes):
        # If the detected object isn't a car / truck, skip it
        if classIDs[i] in [2, 5, 7]:
            car_boxes.append(box)
    return car_boxes

# comapre two frames
def compare_images(frame, next_frame):
    roi_current_frame=frame[203:251,194:269]
    roi_next_frame=next_frame[203:251,194:269]
    simlarityIndex = ssim( roi_current_frame, roi_next_frame,multichannel=True)
    if simlarityIndex>0.2:
        return True
    else:
        return False

def parking_detection(image):
    confidence_threshold=0.5
    threshold_t=0.3

# construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities
    # grab image spatial dimensions
    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # uncomment to see timing information on YOLO
    # print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
    	# loop over each of the detections
    	for detection in output:
    		# extract the class ID and confidence (i.e., probability) of
    		# the current object detection
    		scores = detection[5:]
    		classID = np.argmax(scores)
    		confidence = scores[classID]

    		# filter out weak predictions by ensuring the detected
    		# probability is greater than the minimum probability
    		if confidence > confidence_threshold:
    			# scale the bounding box coordinates back relative to the
    			# size of the image, keeping in mind that YOLO actually
    			# returns the center (x, y)-coordinates of the bounding
    			# box followed by the boxes' width and height
    			box = detection[0:4] * np.array([W, H, W, H])
    			(centerX, centerY, width, height) = box.astype("int")

    			# use the center (x, y)-coordinates to derive the top and
    			# and left corner of the bounding box
    			x = int(centerX - (width / 2))
    			y = int(centerY - (height / 2))

    			# update our list of bounding box coordinates, confidences,
    			# and class IDs
    			boxes.append([x, y, int(width), int(height)])
    			confidences.append(float(confidence))
    			classIDs.append(classID)
    car_boxes = get_car_boxes(boxes, classIDs)
    # the bounding box of the marked parking spot
    target_parking_lot=[194,203,75,48]
    font = cv2.FONT_HERSHEY_DUPLEX
    # setting a range for detected bounding box variation
    box_range=[]
    for i in car_boxes:
        box_range.append(i[0])
    if abs(min(box_range, key=lambda v: abs(target_parking_lot[0]-v)) - target_parking_lot[0]) < 6:
        cv2.putText(image, "occupid", (194 + 6, 265 - 6), font, 0.3, (255, 255, 255))
        cv2.rectangle(image, (194, 203), (194+75, 203+48), (255, 0, 0), 1)
        cv2.imwrite('image_new.jpg',image)
        return True
    else:
        return False


# The main section that checks the input arguments and start downloading the weights and intializing the yolov3
if __name__ == "__main__":
    ap = argparse.ArgumentParser("Processing the parking detection request")
    ap.add_argument('function', help="function name")
    ap.add_argument('timestamp1', help="first timestamp")
    ap.add_argument("timestamp2", help="second timestamp", nargs='?')
    ap.add_argument("-o", "--output",  help="output file name")
    args = vars(ap.parse_args())
    if args['output']:
        output_file_name = args['output']
    else:
        output_file_name = None
    args = sys.argv[1:]
    if (args[0] == 'compare' or args[0] == 'analyze') and len(args) < 3:
      print("error: must specify TWO timestamps for comparison/analyze ")
      sys.exit(1)
    # downloading the YOLO weights to yolo-coco folder
    cache.get_yolo_file()
    model_name = "yolo-coco"
    # load the COCO class labels that our YOLO model was trained on
    labelsPath = os.path.sep.join([model_name, "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
        dtype="uint8")

    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([model_name, "yolov3.weights"])
    configPath = os.path.sep.join([model_name, "yolov3.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # dictionary to save the output as jason file
    json_data = {}

    if args[0] == 'detect':
      print("detect is running")
      if parking_detection(cv2.imread(cache.get_image(args[1]))):
          print("The parking Spot is occupied")
          json_data["detect for : " + args[1]] = " The parking Spot is occupied "
      else:
          print("The parking Spot is not occupied")
          json_data["detect for : " + args[1]] = " The parking Spot is not occupied "
    elif args[0] == 'compare':
      print("compare is running")
      image1 = cv2.imread(cache.get_image(args[1]))
      image2 = cv2.imread(cache.get_image(args[2]))
      if compare_images(image1, image2):
          print("The same car detected")
          json_data["compare of : "+ args[1] + " and " + args[2]] = " The same car detected "
      else:
          if parking_detection(image2):
              print("new car is detected")
              json_data["compare of : "+ args[1] + " and " + args[2]] = " new car is detected "
          else:
              print("The parking Spot is not occupied")
              json_data["compare of : "+ args[1] + " and " + args[2]] = " The parking Spot is not occupied "
    else:
      print("analyze is running")
      sampling_rate = 4
      any_car_detected = 0
      index_file_list = cache.get_index_file()
      first_index = index_file_list.index(args[1])
      last_index = index_file_list.index(args[2])
      # This loop is used to detect the first time that a car is parked then it breaks
      for j in range(first_index, last_index):
          if parking_detection(cv2.imread(cache.get_image(index_file_list[j]))):
              any_car_detected = 1
              occupied = 1
              print(" One car is detected at ts = ", index_file_list[j])
              json_data["analyse of : "+ index_file_list[j]] = " One car is detected at ts = " + str(index_file_list[j])
              detected_car_first_ts = index_file_list[j]
             # This loop is used to check the parking sopt between each timestamp and its next timestamp
              for i in range(j, last_index):
                  if not occupied:
                      if parking_detection(cv2.imread(cache.get_image(index_file_list[i]))):
                          print("new car is detected at ts = ", index_file_list[i])
                          json_data["analyse of : " + index_file_list[i]] = " new car is detected at ts = " + str(index_file_list[i])
                          occupied = 1
                          detected_car_first_ts = index_file_list[i]
                # if a car leaving the parking waits for more than few sampling rate it will be detected as a new car and we ignore the parking time less than one mintue
                  elif not compare_images(cv2.imread(cache.get_image(index_file_list[i])), cv2.imread(cache.get_image(index_file_list[i+1]))) and int(index_file_list[i+1]) - int(detected_car_first_ts) > sampling_rate*15:
                      detected_car_last_ts =  index_file_list[i+1]
                      print("found car at {}. parked until {} ({} minutes).".format(detected_car_first_ts, detected_car_last_ts, round((int(detected_car_last_ts) - int(detected_car_first_ts))/60)))
                      json_data["analyse of : " + str(detected_car_first_ts) + " and " + str(detected_car_last_ts)] = " found car at: " + str(detected_car_first_ts) + " parked until " + str(detected_car_last_ts) + "for" + str(round((int(index_file_list[i]) - int(detected_car_first_ts))/60)) + " minutes."
                      if parking_detection(cv2.imread(cache.get_image(index_file_list[i+1]))):
                          print("new car is detected at ts", index_file_list[i+1])
                          json_data["analyse of : " + index_file_list[i+1]] = " New car is detected at ts = " + str(index_file_list[i+1])
                          detected_car_first_ts = index_file_list[i+1]
                          occupied = 1
                      else:
                          print("The parking Spot is not occupied")
                          json_data["analyse of : "+ index_file_list[i+1]] = " The parking Spot is not occupied at: " + str(index_file_list[i+1])
                          occupied= 0
                  if i == last_index-1 and any_car_detected == 1:
                      print("found car at {}. parked until {} ({} minutes).".format(detected_car_first_ts, index_file_list[i], round((int(index_file_list[i]) - int(detected_car_first_ts))/60)))
                      json_data["analyse of : "+ str(detected_car_first_ts) + " and " + str(index_file_list[i])] = " found car at: " + str(detected_car_first_ts) + " parked until " + str(index_file_list[i]) + " for " + str(round((int(index_file_list[i]) - int(detected_car_first_ts))/60)) + " minutes."

                      print("This car was parked here till the last ts: ", index_file_list[i])
              break
          else:
              print(" No car is detected yet ... ")
              print(" current frame: ", index_file_list[j])
              json_data["analyse of : " + index_file_list[j]] = " No car is detected yet, current frame: " + str(index_file_list[j])
      if not any_car_detected:
          print("NO car detected during this period")
if output_file_name:
    with open(output_file_name, "w") as write_file:
        json.dump(json_data, write_file)
