"""
This was modified from the work detailed in [https://debuggercafe.com/instance-segmentation-with-pytorch-and-mask-r-cnn/],
to work with video files and working with the required file and output requirements

also used is a pre-trained MaskRCNN model [https://pytorch.org/vision/main/models/mask_rcnn.html],
on the COCO dataset [https://cocodataset.org/#home]
"""
# from google.colab import files

import torch
import torchvision
import cv2 as cv
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

from torchvision.transforms import transforms as transforms

import utils
from utils import *

# some starting constants
# dataset_path = "/content/drive/MyDrive/COMP_SCI/Vision/Dataset/Train"
dataset_path = "Dataset/Train"

# initialize the model
maskRCNN_model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
                                                                    progress=True,
                                                                    num_classes=91)
# set the computation device
# load the modle on to the computation device and set to eval mode
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
maskRCNN_model.to(device).eval()

HEIGHT = 720
WIDTH = 1280

img_transform = transforms.Compose([
    transforms.ToTensor()
])


def get_outputs(image, model, threshold, cpu_option=False):
    """
    modified from https://debuggercafe.com/instance-segmentation-with-pytorch-and-mask-r-cnn/
    applies the pretrained MaskRCNN model on an input image. [https://pytorch.org/vision/main/models/mask_rcnn.html]
    :param image:
    :param model:
    :param threshold:
    :return:
    """
    if not cpu_option:
        with torch.no_grad():
            # forward pass of the image through the model
            outputs = model(image)
    else:
        model.to("cpu")
        with torch.no_grad():
            outputs = maskRCNN_model(image.to("cpu"))

    # get all the scores
    scores = list(outputs[0]['scores'].detach().cpu().numpy())

    # get the classes labels
    labels = [coco_names[i] for i in outputs[0]['labels']]
    # index of those scores which are above a certain threshold
    person_preds_inidices = [i for i, item in enumerate(labels) if item == 'person']

    # index of those labels wich are labeled people
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]

    # preds_inidices = list(set(thresholded_preds_inidices).intersection(set(person_preds_inidices)))
    preds_inidices = [scores.index(i) for i in scores if scores.index(i) in person_preds_inidices][:1]
    preds_count = len(list(preds_inidices))

    # get the masks
    masks = (outputs[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    # discard masks for objects which are below threshold
    masks = masks[:preds_count]

    # get the bounding boxes, in (x1, y1), (x2, y2) format
    boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in outputs[0]['boxes'].detach().cpu()]
    # discard bounding boxes below threshold value
    boxes = boxes[:preds_count]

    return masks, boxes, labels


def draw_segmentation_map(image, masks, boxes, labels, square=True):
    # convert the original PIL image into NumPy format
    original_image = np.array(image)

    images = []
    for i in range(len(masks)):
        if image.shape[:2] != masks[i].shape:
            return None
        image = original_image.copy()
        # convert from RGN to OpenCV BGR format
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR).astype("uint8")

        # apply mask on the image

        masked_image = cv.bitwise_and(image, image, mask=np.uint8(masks[i] * 255).astype("uint8"))

        # crop by the bounding boxes around the objects
        image = image[boxes[i][0][1]:boxes[i][1][1], boxes[i][0][0]:boxes[i][1][0]]
        masked_image = masked_image[boxes[i][0][1]:boxes[i][1][1], boxes[i][0][0]:boxes[i][1][0]]

        # fix the aspect ratio to a square for each
        if square:
            image = squareAspect(image, height_len=700, side_len=700)
            masked_image = squareAspect(masked_image, height_len=700, side_len=700)

        images.append(cv.cvtColor(masked_image, cv.COLOR_BGR2RGB))
        """cv.imwrite(f"test_{i}.jpg", cv.cvtColor(masked_image, cv.COLOR_BGR2RGB))
    cv.imwrite(f"test_referance.jpg", cv.cvtColor(original_image, cv.COLOR_BGR2RGB))"""
    return images


def runPersonExtraction(image, threshold=0.5, cpu=False, square=True):
    orig_image = image.copy()

    # transform the image
    image = img_transform(image)

    # add a batch dimension
    image = image.unsqueeze(0).to(device)
    masks, boxes, labels = get_outputs(image, maskRCNN_model, threshold, cpu)
    return draw_segmentation_map(orig_image, masks, boxes, labels, square=square), masks, boxes, labels


def processVideos(ds_path, results_path):
    img_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    c = 0
    for file_loc in os.listdir(ds_path):
        for video_file in os.listdir(os.path.join(ds_path, file_loc)):
            img_gen = videoImgGenerator(video_path=os.path.join(ds_path, file_loc, video_file),
                                        framerate=15)

            video_name = video_file.split(".")[0].replace(" ", "_")
            patch_save_path = os.path.join(results_path, "1_1", file_loc, video_name)
            frame_save_path = os.path.join("Frames", file_loc, video_name)

            utils.createIfNotExist(patch_save_path)
            utils.createIfNotExist(frame_save_path)

            second = 0
            for img in img_gen:
                cv.cvtColor(img, cv.COLOR_RGB2BGR)
                cv.imwrite(os.path.join(frame_save_path, f"{file_loc[0]}_{video_name}_{second}.jpg"),
                           cv.cvtColor(img, cv.COLOR_RGB2BGR))

                images, _, _, _ = runPersonExtraction(img, threshold=0.99)

                if images is not None:
                    for i in range(len(images)):
                        base = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
                        x_pos = (720 - 700) // 2
                        y_pos = (1280 - 700) // 2
                        base[x_pos:x_pos + 700, y_pos:y_pos + 700, :] = images[i]

                        cv.imwrite(os.path.join(patch_save_path, f"{file_loc[0]}_{video_name}_{second}_{i}.jpg"), base)

                    print(f"video: {video_file}, s:{second}")

                second += 1

    print(f"processed {c} images")


def q11():
    processVideos(ds_path=os.path.join(".", "Dataset", "Train"), results_path=os.path.join(".", "Results", "1_1"))
    utils.displayRandom(50, path=os.path.join(".", "Results", "1_1"))

if __name__ == '__main__':
    q11()