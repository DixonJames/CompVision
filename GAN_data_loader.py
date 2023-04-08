"""
This code parts that were modified from the CycleGAN tutorial
at: [https://www.tensorflow.org/tutorials/generative/cyclegan?fbclid=IwAR1-7VDGw93g9p7ohYh8MUqNrTaHn_J3etrsQe4AW0MMTXnCc2-loZY77bo]

it also uses the pix2pix model from [https://github.com/tensorflow/examples]
wich has been imported and used in the local pix2pix file in this project

@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkss},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}

"""
import os
import cv2 as cv
from sklearn.model_selection import train_test_split
import tensorflow as tf


def getDatasets(path, split=0.2):
    label_key = {"Game": 0, "Movie": 1}

    datasets = []
    for class_label in os.listdir(path):
        parts = []
        for folder in os.listdir(os.path.join(path, class_label)):
            for image_file in os.listdir(os.path.join(path, class_label, folder)):
                vector = os.path.join(path, class_label, folder, image_file)

                parts.append(vector)
        datasets.append(parts)

    game_train_filepaths, game_test_filepaths = train_test_split(datasets[0], test_size=split)
    movie_train_filepaths, movie_test_filepaths = train_test_split(datasets[1], test_size=split)

    return (game_train_filepaths, game_test_filepaths), (movie_train_filepaths, movie_test_filepaths)


def random_crop(image):
    cropped_image = tf.image.random_crop(
        image, size=[256, 256, 3])

    return cropped_image


# normalizing the images to [-1, 1]
def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


def random_jitter(image):
    # resizing to 286 x 286 x 3
    image = tf.image.resize(image, [286, 286],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # randomly cropping to 256 x 256 x 3
    image = random_crop(image)

    # random mirroring
    image = tf.image.random_flip_left_right(image)

    return image


def preprocess_image_train(image):
    image = random_jitter(image)
    image = normalize(image)
    return image


def preprocess_image_test(image):
    image = normalize(image)
    return image


def loadTrainImage(filepath):
    img = cv.imread(filepath)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [256, 256])

    img = preprocess_image_train(img)
    return img


def loadTestImage(input, resize=True, convert=False, reseize_dim=256):
    if not convert:
        img = cv.imread(input)
    else:
        img = input
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.image.convert_image_dtype(img, tf.float32)
    if resize:
        img = tf.image.resize(img, [reseize_dim, reseize_dim])

    img = preprocess_image_test(img)
    return img


def tfDatasetGen(paths, train=True, batch_size=32):
    """
    modified from: [https://stackoverflow.com/questions/70745528/fit-tf-data-dataset-with-from-generator]
    :return:
    """

    def imgGen(paths, train):
        for path in paths:
            if train == True:
                yield loadTrainImage(path)
            else:
                yield loadTestImage(path)

    image_dataset = tf.data.Dataset.from_generator(
        lambda: imgGen(paths, train),
        output_types=tf.float32,
        output_shapes=(tf.TensorShape([256, 256, 3]))
    )

    image_dataset = image_dataset.shuffle(buffer_size=len(paths))
    image_dataset = image_dataset.batch(batch_size)

    return image_dataset


def DataSetloaders(path, batch_size=32, lim_size=0):
    (game_train_filepaths, game_test_filepaths), (movie_train_filepaths, movie_test_filepaths) = getDatasets(
        path)

    if lim_size != 0:
        game_train_filepaths, game_test_filepaths, movie_train_filepaths, movie_test_filepaths = \
            game_train_filepaths[:lim_size], game_test_filepaths[:lim_size], movie_train_filepaths[
                                                                             :lim_size], movie_test_filepaths[:lim_size]
    game_train_ds = tfDatasetGen(game_train_filepaths, train=True, batch_size=batch_size)
    game_test_ds = tfDatasetGen(game_train_filepaths, train=False, batch_size=batch_size)

    movie_train_ds = tfDatasetGen(movie_train_filepaths, train=True, batch_size=batch_size)
    movie_test_ds = tfDatasetGen(movie_test_filepaths, train=False, batch_size=batch_size)

    return game_train_ds, game_test_ds, movie_train_ds, movie_test_ds


def tensor2Image(tensor):
    return cv.cvtColor((tensor.numpy() * 0.5 + 0.5) * 255, cv.COLOR_RGB2BGR)


if __name__ == '__main__':
    game_train_filepaths, game_test_filepaths, movie_train_filepaths, movie_test_filepaths = DataSetloaders(
        "Results/1_1_edited", 32)
