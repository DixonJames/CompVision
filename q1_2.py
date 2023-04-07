"""
This work is modified from:
1: [https://github.com/quanhua92/human-pose-estimation-opencv/blob/master/openpose.py]
2: [https://github.com/opencv/opencv/blob/master/samples/dnn/openpose.py]

with intended use of the OpenPose model [3] that is pretrained on the COCO dataset [4] within the OpenCV library [5]

3: [https://github.com/CMU-Perceptual-Computing-Lab/openpose.git]
4: [https://cocodataset.org/#home]
5: [https://github.com/opencv/opencv/blob/master/samples/dnn/openpose.py]

before use:
    ensure that pose_deploy_linevec.prototxt & pose_iter_440000.caffemodel are downloaded an in the same
    directory.

classification is modified from [https://medium.com/analytics-vidhya/a-simple-neural-network-classifier-using-pytorch-from-scratch-7ebb477422d2]
"""
import cv2 as cv
import numpy as np
import argparse
import os
import json
from utils import *

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pandas as pd

model_weights_path = "pose_deploy_linevec.prototxt"
model_architecture_path = "pose_iter_440000.caffemodel"
points_save_path = "points.json"

BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
              "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
              ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
              ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
              ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

body_parts_key = {"half": 0, "head": 1, "sit": 2, "stand": 3, "other": 4}

inWidth = 1280
inHeight = 720

inScale = 0.003922
threshold = 0.01

pose_model = cv.dnn.readNet(cv.samples.findFile(model_weights_path), cv.samples.findFile(model_architecture_path))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


def createLabels():
    """
    creates labels.json file
    :return:
    """
    label_path = os.path.join("labels.json")
    if not os.path.exists(label_path):
        with open(label_path, "w") as file:
            json.dump({}, file)

    with open(label_path, 'r') as file:
        label_json_file = json.load(file)

    for file_loc in os.listdir(os.path.join(".", "mydataset")):
        for img_loc in os.listdir(os.path.join(".", "mydataset", file_loc)):
            label_json_file[img_loc] = file_loc

    with open(label_path, 'w') as file:
        json.dump(label_json_file, file)


def poseImage(image, points):
    """
    creates the image with skeleton overlay
    :param image:
    :param points:
    :return:
    """
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]

        assert (partFrom in BODY_PARTS)
        assert (partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if (points[idFrom] == (0, 0)) or (points[idTo] == (0, 0)):
            continue

        if points[idFrom] and points[idTo]:
            cv.line(image, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(image, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(image, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
    return image


def savePoints(point, name, save_path):
    """
    saves the points generated to points.json file
    :param point:
    :param name:
    :param save_path:
    :return:
    """
    if not os.path.exists(save_path):
        with open(save_path, "w") as file:
            json.dump({}, file)

    with open(save_path, 'r') as file:
        json_file = json.load(file)

    json_file[name] = point

    with open(save_path, 'w') as file:
        json.dump(json_file, file)


def poseEstimation():
    """
    goes though all the reults of part 1_1
    finds the skeleton points of each
    :return:
    """
    p0 = os.path.join(".", "Results", "1_1_edited")
    folders = os.listdir(p0)
    folders.reverse()
    for file_loc in folders:
        print(file_loc)

        p1 = os.path.join(p0, file_loc)
        for video_cap_file in os.listdir(p1):
            print(video_cap_file)

            p2 = os.path.join(p1, video_cap_file)
            for image_file in os.listdir(p2):

                save_path = os.path.join(".", "Results", "1_2", file_loc, video_cap_file)
                image = cv.imread(os.path.join(p2, image_file), cv.IMREAD_COLOR)

                frameWidth, frameHeight = image.shape[1], image.shape[0]

                reformatted_image = cv.dnn.blobFromImage(image, inScale, (inWidth, inHeight),
                                                         (0, 0, 0), swapRB=False, crop=False)
                pose_model.setInput(reformatted_image)
                out = pose_model.forward()

                assert (len(BODY_PARTS) <= out.shape[1])

                points = []

                result = BODY_PARTS.copy()
                for i in range(len(BODY_PARTS)):
                    heatMap = out[0, i, :, :]
                    _, conf, _, point = cv.minMaxLoc(heatMap)
                    x = (frameWidth * point[0]) / out.shape[3]
                    y = (frameHeight * point[1]) / out.shape[2]

                    points.append((int(x), int(y)) if conf > threshold else (0, 0))
                    result[(list(BODY_PARTS.keys()))[i]] = (int(x), int(y))

                labeled = poseImage(image, points)
                save_name = file_loc[0] + "_" + image_file
                saveIMG(labeled, save_name, save_path)

                savePoints(result, save_name, points_save_path)


def load_parts(labels_path, vectors_path):
    """
    conversts the json file contents to vector for use in model

    :param labels_path: labels.json path
    :param vectors_path: points.json path
    :return: vectors of points and labels
    """

    labels_results = []
    vectors_results = []
    vec_to_filename = {}
    with open(labels_path, 'r') as file:
        labels = json.load(file)
    with open(vectors_path, 'r') as file:
        vectors = json.load(file)

    for img_name in list(vectors.keys()):

        vec_2d = list(vectors[img_name].values())
        if img_name[0].isdigit():
            img_name = "G_" + img_name
        vec = []
        for (x, y) in vec_2d:
            vec.append(x)
            vec.append(y)

        if img_name in list(labels.keys()):
            label = body_parts_key[labels[img_name]]
        else:
            label = 5

        labels_results.append(label)
        vectors_results.append(vec)
        vec_to_filename[str(vec)] = img_name

    return vectors_results, labels_results, vec_to_filename


class ActionDataset(Dataset):
    def __init__(self, labels, vectors):
        self.labels = labels
        self.vectors = vectors

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        vector = torch.Tensor(self.vectors[index])
        return label, vector


class ActionNetwork(nn.Module):
    def __init__(self):
        super(ActionNetwork, self).__init__()
        self.l_1 = nn.Linear(38, 64)
        self.r_1 = nn.ReLU()
        self.l_2 = nn.Linear(64, 32)
        self.r_2 = nn.ReLU()
        self.l_3 = nn.Linear(32, 5)

    def forward(self, x):
        x = self.l_1(x)
        x = self.r_1(x)
        x = self.l_2(x)
        x = self.r_2(x)
        x = self.l_3(x)
        return x


def getDataset(labels_path, vectors_path):
    """
    turns the labels and points json into a train test dataset
    also gives a doct from vectors to the orginal image file name
    """
    all_vectors, all_labels, vector_key = load_parts(labels_path, vectors_path)

    df = pd.DataFrame(columns=["vectors", "labels"])
    df["vectors"] = all_vectors
    df["labels"] = all_labels

    second_highest = sorted(df["labels"].value_counts())[-2]
    selected_label_0 = df[df["labels"] == 0].sample(second_highest, replace=False)
    df = df[df['labels'] != 0]

    df = pd.concat([df, selected_label_0], axis=0)
    df.reset_index(drop=True, inplace=True)

    all_vectors, all_labels = list(df["vectors"]), list(df["labels"])

    train_data, test_data, train_labels, test_labels = train_test_split(all_vectors, all_labels,
                                                                        test_size=0.2, random_state=1)

    batch_size = 64

    train_dataset = ActionDataset(labels=train_labels, vectors=train_data)
    train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = ActionDataset(labels=test_labels, vectors=test_data)
    test_dataset_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset_loader, test_dataset_loader, vector_key


def testModel(network, test_dataset_loader, get_predictions=True):
    """
    test model in test dataset
    :return accuracey and wether to continue trianing
    """
    # test network
    correct = 0
    total = 0
    predictions = {}
    with torch.no_grad():
        for labels, inputs in test_dataset_loader:
            # convert data to device
            inputs, labels = inputs.to(device).float(), labels.to(device)

            outputs = network(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.squeeze()).sum().item()

            if get_predictions:
                for prediction, vec in list(zip((predicted.to('cpu').numpy()), (inputs.to('cpu').numpy().tolist()))):
                    predictions[str([int(i) for i in vec])] = prediction

    return 100 * correct / total, predictions


def trainModel(network, optimizer, loss_func, train_dataset_loader, test_dataset_loader, accuracy_stagnation=5):
    top_test_acc = 0.0
    test_stagnation = 0
    train_indicator = True
    current_epoch = 0
    # train network
    while train_indicator:
        epoch_train_loss = 0.0
        for labels, inputs in train_dataset_loader:
            # convert data to device
            inputs, labels = inputs.to(device).float(), labels.to(device)

            # remove previsous grad
            optimizer.zero_grad()

            # run network on data
            outputs = network(inputs)

            # calculate the crossEntropy Loss
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            # add to epochs cumulative loss
            epoch_train_loss += loss.item()

        # print(f'Epoch: {current_epoch + 1}  \t\t\t Training Loss:{epoch_train_loss / len(train_dataset_loader)}')
        current_epoch += 1

        # test network
        correct = 0
        total = 0
        with torch.no_grad():
            for labels, inputs in test_dataset_loader:
                # convert data to device
                inputs, labels = inputs.to(device).float(), labels.to(device)

                outputs = network(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.squeeze()).sum().item()

        test_acc, _ = testModel(network=network, test_dataset_loader=test_dataset_loader, get_predictions=True)

        if test_acc < top_test_acc:
            if test_stagnation == accuracy_stagnation:
                train_indicator = False
            test_stagnation += 1
        else:
            top_test_acc = test_acc
            test_stagnation = 0

    return top_test_acc, network


def predictTest(model, testIterator, vector_key):
    _, predictions = testModel(network=model, test_dataset_loader=testIterator)
    results = pd.DataFrame(columns=["img", "prediction"])
    results["img"] = [vector_key[k] for k in predictions.keys()]
    results["prediction"] = predictions.values()

    classes_prediction_selection = []
    for class_label in range(5):
        selected_label = list((results[results["prediction"] == class_label].sample(10, replace=False))["img"])
        classes_prediction_selection.append(selected_label)

    return classes_prediction_selection


def runPredictions():
    # create the points.json file of each of the joint locations (and optionally the images displaying them)

    poseEstimation()

    # creates the labels.json file (submitted with CW)
    # this tuned on my manually labeled dataset of the images into labels to train on
    # this doesn't have to be run when part of the submission as labels.json is already included in the submission.

    """createLabels()"""

    current_epoch = 0

    # create the dataset to train the model
    """
    train_dataset_loader, test_dataset_loader, vector_key = getDataset("labels.json", "points.json")
    """

    # create network
    """
    network = ActionNetwork()
    network.to(device)
    """

    # get loss function and optimiser
    """
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=0.002, betas=(0.9, 0.999))
    """

    # train model
    """
    test_acc, model = trainModel(network, optimizer, loss_func,train_dataset_loader, test_dataset_loader, accuracy_stagnation=14)
    """

    # predict all test set
    """
    predictions = predictTest(network, test_dataset_loader, vector_key)
    return predictions
    """


if __name__ == '__main__':
    predictions = runPredictions()
