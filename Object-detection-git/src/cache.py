import os
import requests
import cv2
import re

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
CACHE_DIR = os.path.join(SCRIPT_PATH, '..', 'cache')
YOLO_DIR = os.path.join(SCRIPT_PATH, '..', 'yolo-coco')
YOLO_WEIGHT='yolov3.weights'
INDEX_FILE = 'index.txt'
YOLO_URL='https://pjreddie.com/media/files/yolov3.weights'
INDEX_URL = 'https://hiring.verkada.com/video/index.txt'
TS_URL = 'https://hiring.verkada.com/video/{}.ts'

def get_cache_dir():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    return CACHE_DIR


def get_index_file():
    cache = get_cache_dir()
    filename = os.path.join(cache, INDEX_FILE)
    if not os.path.exists(filename):
        download_file(INDEX_URL, filename)
    return [re.search(r'.*(?=\.)', line.strip()).group() for line in open(filename)]


def get_image(timestamp):
    '''
    downloads a ts file and writes the first frame to the cache as a jpeg.

    timestamp is an integer (seconds since unix epoch)
    '''
    video_URL= TS_URL.format(timestamp)
    video_filename= video_URL[video_URL.rfind("/")+1:]
    download_file(video_URL, video_filename)
    VIDEO_SOURCE =video_filename
    video_capture = cv2.VideoCapture(VIDEO_SOURCE)
    success, frame = video_capture.read()
    cv2.imwrite(os.path.join(CACHE_DIR ,"frame{}.jpg".format(timestamp)), frame)
    os.remove(video_filename)
    return os.path.join(CACHE_DIR ,"frame{}.jpg".format(timestamp))


def download_file(url, filename):
    '''
    downloads the contents of the provided url to a local file
    '''
    contents = requests.get(url).content
    with open(filename, 'wb') as f:
        f.write(contents)


def get_yolo_dir():
    if not os.path.exists(YOLO_DIR):
        os.makedirs(YOLO_DIR)
    return YOLO_DIR


def get_yolo_file():
    yolo_weight = get_yolo_dir()
    filename = os.path.join(yolo_weight, YOLO_WEIGHT)
    if not os.path.exists(filename):
        download_file(YOLO_URL, filename)
