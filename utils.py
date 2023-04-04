# from google.colab import files
import cv2

import numpy as np
from PIL import Image
from IPython import display
import os

d = display.display(None, display_id=True)

coco_names = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
              'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
              'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
              'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
              'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A',
              'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
              'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A',
              'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
              'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase',
              'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def download(filename):
    files.download(filename)


def folderDownload(folder_path):
    # !zip -r ./CompressedFiles1.zip ./folder_path/
    files.download("CompressedFiles1.zip")


def arrayToImg(array, dispaly_img=False):
    img = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    if dispaly_img:
        d.update(img)
    return img


def videoImgGenerator(video_path, framerate=30):
    cap = cv2.VideoCapture(video_path)
    ret = True
    c = 0
    while True:
        # Read a frame from the video
        while c < framerate:
            c += 1
            ret, frame = cap.read()
        else:
            c = 0
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            yield rgb_frame

    cap.release()
    cv2.destroyAllWindows()


def squareAspect(image, side_len=512):
    # modified from [https://note.nkmk.me/en/python-pillow-add-margin-expand-canvas/]
    h, w, _ = image.shape

    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if w > h:
        result = Image.new(image.mode, (w, w), (0, 0, 0))
        result.paste(image, (0, (w - h) // 2))
        image = result
    elif h > w:
        result = Image.new(image.mode, (h, h), (0, 0, 0))
        result.paste(image, ((h - w) // 2, 0))
        image = result
    image = image.resize((side_len, side_len))

    bgr_nparray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    return cv2.cvtColor(bgr_nparray, cv2.COLOR_BGR2RGB)

