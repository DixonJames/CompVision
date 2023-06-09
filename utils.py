# from google.colab import files
import cv2 as cv
import random
import numpy as np
from PIL import Image
from IPython import display
import os
import matplotlib.pyplot as plt

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
    img = cv.cvtColor(array, cv.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    if dispaly_img:
        d.update(img)
    return img


def videoImgGenerator(video_path, framerate=30, batches=32):
    cap = cv.VideoCapture(video_path)
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
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            yield rgb_frame

    cap.release()
    cv.destroyAllWindows()


def squareAspect(image, height_len, side_len):
    # modified from [https://note.nkmk.me/en/python-pillow-add-margin-expand-canvas/]
    h, w, _ = image.shape

    image = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))

    if w > h:
        result = Image.new(image.mode, (w, w), (0, 0, 0))
        result.paste(image, (0, (w - h) // 2))
        image = result
    elif h > w:
        result = Image.new(image.mode, (h, h), (0, 0, 0))
        result.paste(image, ((h - w) // 2, 0))
        image = result
    image = image.resize((height_len, side_len))

    return np.array(image)


def createIfNotExist(path):
    folders = path.split('/')
    current_path = ""
    for f in folders:
        current_path = os.path.join(current_path, f)
        if not os.path.exists(current_path):
            os.mkdir(current_path)


def getRandomImges(n, path):
    all_imgs = []
    for file in os.listdir(path):
        for img_path in os.listdir(os.path.join(path, file)):
            if img_path.split(".")[-1] == "jpg":
                all_imgs.append(os.path.join(path, file, img_path))
            else:
                for img in os.listdir(os.path.join(path, file, img_path)):
                    all_imgs.append(os.path.join(path, file, img_path, img))

    return random.sample(all_imgs, n)


def displayRandom(n, path):
    random_paths = getRandomImges(n, path)
    for i, img_path in enumerate(random_paths):
        plt.figure(figsize=(12, 12))
        img = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2RGB)
        plt.title(f"Random image {i}")
        plt.imshow(img)
        plt.axis('off')
        plt.show()


def displayPicture(path, title):
    plt.figure(figsize=(12, 12))
    img = cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB)
    plt.title(f"{title}")
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def displayImg(img):
    plt.figure(figsize=(12, 12))
    for i in range(1):
        plt.subplot(1, 1, i + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.show()


def saveIMG(image, name, save_path):
    createIfNotExist(save_path)
    cv.imwrite(os.path.join(save_path, name), image)

def find_files(filename, search_path):
    """
    modified from [https://www.tutorialspoint.com/file-searching-using-python]
    """
    result = []
    for root, dir, files in os.walk(search_path):
        if filename in files:
            result.append(os.path.join(root, filename))
    return result

def frame2Vid(path, title):
    """
    takes a path fo images, orders by frame number and creates a video
    :param path:
    :return:
    """
    output_video = f"{path}/{title}.mp4"

    fps = 25
    frame_size = (480, 360)
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video_writer = cv.VideoWriter(output_video, fourcc, fps, frame_size)

    frame_paths = []
    for filename in sorted(os.listdir(path)):
        if filename.endswith(".jpg"):
            filepath = os.path.join(path, filename)
            frame_paths.append(filepath)

    frame_paths.sort(key=lambda x: int(x.split(".")[0].split("_")[-1]))

    for pth in frame_paths:
        frame = cv.imread(pth)
        video_writer.write(frame)

    video_writer.release()
