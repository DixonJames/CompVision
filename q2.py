"""
This code was modified from the CycleGAN tutorial
at: [https://www.tensorflow.org/tutorials/generative/cyclegan?fbclid=IwAR1-7VDGw93g9p7ohYh8MUqNrTaHn_J3etrsQe4AW0MMTXnCc2-loZY77bo]

it also uses the pix2pix model from [https://github.com/tensorflow/examples]
wich has been imported and used in the local pix2pix file in this project
"""
import json

import tensorflow as tf

tf.compat.v1.enable_eager_execution()
# import pix2pix
from examples.tensorflow_examples.models.pix2pix import pix2pix
import time
import json
import os
import scipy
import matplotlib.pyplot as plt
from IPython.display import clear_output
import GAN_data_loader
import cv2 as cv
import math
import utils
import numpy as np
# from q1_1 import *

inception_model = tf.keras.applications.InceptionV3(include_top=False,
                                                            weights="imagenet",
                                                            pooling='avg')

from PIL import Image

BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3

lim_ds_size = 64

"""imgs = [sample_movie[0], to_game[0], sample_game[0], to_movie[0]]
title = ['Horse', 'To Zebra', 'Zebra', 'To Horse']

for img, title in zip(imgs, title):
        cv.imwrite(os.path.join(f"{title}.jpg"), GAN_data_loader.tensor2Image(img))"""


class CycleGAN:
    def __init__(self, BUFFER_SIZE=1000, BATCH_SIZE=16, IMG_WIDTH=256, IMG_HEIGHT=256, OUTPUT_CHANNELS=3, LAMBDA=10,
                 EPOCHS=10, lim_ds_size=0, checkpoint_path="./checkpoints/train", training_ds="Results/1_1",
                 loss_path="./", FID_save_path="./"):

        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_HEIGHT = IMG_HEIGHT
        self.OUTPUT_CHANNELS = OUTPUT_CHANNELS
        self.LAMBDA = LAMBDA
        self.EPOCHS = EPOCHS
        self.lim_ds_size = lim_ds_size
        self.loss_save_path = loss_path

        self.FID_save_path = FID_save_path
        self.FID_scores = {"game":[], "movie":[]}

        self.sample_img = None

        self.checkpoint_path = checkpoint_path

        self.game_train_loader, self.game_test_loader, self.movie_train_loader, self.movie_test_loader = GAN_data_loader.DataSetloaders(
            training_ds, batch_size=self.BATCH_SIZE, lim_size=self.lim_ds_size)

        # cv.imwrite(os.path.join(f"movie_train.jpg"), GAN_data_loader.tensor2Image(sample_movie[0]))

        # Generator G translates X -> Y, movies to games
        # Generator F translates Y -> X, games to movies

        self.generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
        self.generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

        self.discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
        self.discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

        self.loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.loss_record = {"gen_g_loss": [], "gen_f_loss": [], "disc_movie_loss": [], "disc_game_loss": [],
                            "total_cycle_loss": []}
        self.ckpt = None
        self.ckpt_manager = None
        self.setupCheckpoints()

        self.loadMostRecentCheckpoint()

    def setupCheckpoints(self):
        self.ckpt = tf.train.Checkpoint(generator_g=self.generator_g,
                                        generator_f=self.generator_f,
                                        discriminator_x=self.discriminator_x,
                                        discriminator_y=self.discriminator_y,
                                        generator_g_optimizer=self.generator_g_optimizer,
                                        generator_f_optimizer=self.generator_f_optimizer,
                                        discriminator_x_optimizer=self.discriminator_x_optimizer,
                                        discriminator_y_optimizer=self.discriminator_y_optimizer)

        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, directory=self.checkpoint_path, max_to_keep=5)

    def save_loss(self):
        with open(self.loss_save_path, "w") as file:
            json.dump(self.loss_record, file)

    def loadMostRecentCheckpoint(self):
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

    def load(self):
        # if a checkpoint exists, restore the latest checkpoint.
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

    def movie2game(self, img_path, img_tensor=None):
        # Generator G translates X -> Y, movies to games
        # Generator F translates Y -> X, games to movies
        if img_path is not None:
            img_tensor = GAN_data_loader.loadTestImage(img_path, resize=True)
        else:
            img_tensor = GAN_data_loader.loadTestImage(img_tensor, resize=True, convert=True)

        expandede_img_tensor = tf.expand_dims(img_tensor, axis=0)
        repeited_img_tensor = tf.repeat(expandede_img_tensor, axis=0, repeats=16)

        preidcted_tensor = self.generator_g(repeited_img_tensor)

        preidcted_tensor = ((preidcted_tensor[0].numpy() + 1) / 2) * 255
        return preidcted_tensor

    def discriminator_loss(self, real, generated):
        real_loss = self.loss_obj(tf.ones_like(real), real)

        generated_loss = self.loss_obj(tf.zeros_like(generated), generated)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss * 0.5

    def generator_loss(self, generated):
        return self.loss_obj(tf.ones_like(generated), generated)

    def calc_cycle_loss(self, real_image, cycled_image):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

        return self.LAMBDA * loss1

    def identity_loss(self, real_image, same_image):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return self.LAMBDA * 0.5 * loss

    def generate_images(self, model, test_input, num, save=True):
        prediction = model(test_input)

        plt.figure(figsize=(12, 12))

        display_list = [test_input[0], prediction[0]]
        title = ['Input Image', 'Predicted Image']

        for i in range(2):
            plt.subplot(1, 2, i + 1)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        plt.show()
        if save:
            img_array = (prediction[0].numpy() + 1) / 2 * 255
            utils.saveIMG(image=cv.cvtColor(img_array, cv.COLOR_RGB2BGR), name=f"{num}_predicted.jpg",
                          save_path=os.path.join("Results", "2"))

    @tf.function
    def train_step(self, real_movie, real_game):
        # x is the movie, y is the game
        # persistent is set to True because the tape is used more than
        # once to calculate the gradients.

        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates X -> Y, movies to games
            # Generator F translates Y -> X, games to movies

            fake_game = self.generator_g(real_movie, training=True)
            cycled_movie = self.generator_f(fake_game, training=True)

            fake_movie = self.generator_f(real_game, training=True)
            cycled_game = self.generator_g(fake_movie, training=True)

            # same_movie and same_game are used for identity loss.
            same_movie = self.generator_f(real_movie, training=True)
            same_game = self.generator_g(real_game, training=True)

            disc_real_movie = self.discriminator_x(real_movie, training=True)
            disc_real_game = self.discriminator_y(real_game, training=True)

            disc_fake_movie = self.discriminator_x(fake_movie, training=True)
            disc_fake_game = self.discriminator_y(fake_game, training=True)

            # calculate the loss
            gen_g_loss = self.generator_loss(disc_fake_game)
            gen_f_loss = self.generator_loss(disc_fake_movie)
            # print(gen_g_loss)

            total_cycle_loss = self.calc_cycle_loss(real_movie, cycled_movie) + self.calc_cycle_loss(real_game,
                                                                                                     cycled_game)

            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = gen_g_loss + total_cycle_loss + self.identity_loss(real_game, same_game)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + self.identity_loss(real_movie, same_movie)

            disc_movie_loss = self.discriminator_loss(disc_real_movie, disc_fake_movie)
            disc_game_loss = self.discriminator_loss(disc_real_game, disc_fake_game)

        # Calculate the gradients for generator and discriminator
        generator_g_gradients = tape.gradient(total_gen_g_loss,
                                              self.generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss,
                                              self.generator_f.trainable_variables)

        discriminator_x_gradients = tape.gradient(disc_movie_loss,
                                                  self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_game_loss,
                                                  self.discriminator_y.trainable_variables)

        # Apply the gradients to the optimizer
        self.generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                                       self.generator_g.trainable_variables))

        self.generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                                       self.generator_f.trainable_variables))

        self.discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                           self.discriminator_x.trainable_variables))

        self.discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                           self.discriminator_y.trainable_variables))

        @tf.function
        def compute_x(x):
            return x

        self.loss_record["gen_g_loss"].append(float)
        self.loss_record["gen_f_loss"].append(float(gen_f_loss))
        self.loss_record["disc_movie_loss"].append(float(disc_movie_loss))
        self.loss_record["disc_game_loss"].append(float(disc_game_loss))
        self.loss_record["total_cycle_loss"].append(float(total_cycle_loss))

    def mainTrain(self):
        start_ind = True
        for epoch in range(self.EPOCHS):
            start = time.time()

            n = 0
            for image_M, image_G in tf.data.Dataset.zip((self.movie_train_loader, self.game_train_loader)):
                if start:
                    self.sample_img = image_M
                    self.generate_images(self.generator_g, self.sample_img, epoch, save=True)

                self.train_step(image_M, image_G)
                # print(self.loss_record)
                if n % 10 == 0:
                    print('.', end='')
                n += 1
            # ckpt_save_path = self.ckpt_manager.save()

            clear_output(wait=True)
            # Using a consistent image (sample_horse) so that the progress of the model
            # is clearly visible.
            self.generate_images(self.generator_g, self.sample_img, epoch, save=True)
            self.FID(epoch)
            self.save_FID()
            if (epoch + 1) % 5 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                    ckpt_save_path))

            # self.save_loss()

            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                               time.time() - start))
        print(self.loss_record)

    def FID(self, epoch_num):
        """
        modified from [https://wandb.ai/ayush-thakur/gan-evaluation/reports/How-to-Evaluate-GANs-using-Frechet-Inception-Distance-FID---Vmlldzo0MTAxOTI]
        :return:
        """
        inception_model = tf.keras.applications.InceptionV3(include_top=False,
                                                            weights="imagenet",
                                                            pooling='avg')

        # Generator G translates X -> Y, movies to games
        # Generator F translates Y -> X, games to movies
        models = [self.generator_f, self.generator_g]
        TestDatasets = [self.game_test_loader, self.movie_test_loader]
        fids = []

        def compute_embeddings(dataloader, count, model=None):
            """
            modified from [https://wandb.ai/ayush-thakur/gan-evaluation/reports/How-to-Evaluate-GANs-using-Frechet-Inception-Distance-FID---Vmlldzo0MTAxOTI]
            :return:
            """

            image_embeddings = []

            for _ in range(count):
                images = next(iter(dataloader))
                if model is not None:
                    images = model(images)
                embeddings = inception_model.predict(images)
                image_embeddings.extend(embeddings)
            return np.array(image_embeddings)

        def calculate_fid(real_embeddings, generated_embeddings):
            """
            modified from [https://wandb.ai/ayush-thakur/gan-evaluation/reports/How-to-Evaluate-GANs-using-Frechet-Inception-Distance-FID---Vmlldzo0MTAxOTI]
            :return:
            """

            # calculate mean and covariance statistics
            mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
            mu2, sigma2 = generated_embeddings.mean(axis=0), np.cov(generated_embeddings, rowvar=False)
            # calculate sum squared difference between means
            ssdiff = np.sum((mu1 - mu2) ** 2.0)
            # calculate sqrt of product between cov
            covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
            # check and correct imaginary numbers from sqrt
            if np.iscomplexobj(covmean):
                covmean = covmean.real
                # calculate score
            fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
            return fid

        for dataloader, model in zip(TestDatasets, models):
            count = 1  # math.ceil(100 / BATCH_SIZE)

            # compute embeddings for real images
            real_image_embeddings = compute_embeddings(dataloader, count)

            # compute embeddings for generated images
            generated_image_embeddings = compute_embeddings(dataloader, count, model=model)

            # real_image_embeddings.shape, generated_image_embeddings.shape

            fids.append(calculate_fid(real_image_embeddings, generated_image_embeddings))

        self.FID_scores["game"].append(fids[0])
        self.FID_scores["movie"].append(fids[1])

    def save_FID(self):
        with open(self.FID_save_path, "w") as file:
            json.dump(self.FID_scores, file)



def q2_2():
    cg = CycleGAN(lim_ds_size=16, BATCH_SIZE=16,
                  checkpoint_path=os.path.join(os.getcwd(), "checkpoints/train"),
                  training_ds="Results/1_1")

    from q1part1 import runPersonExtraction

    # for now just a few random game frames
    frames = utils.videoImgGenerator("Dataset/Test/Test Movie.mp4", framerate=0)
    frame_num = 0
    for frame in frames:
        image = frame
        # convert to patch
        patch_images, masks, boxes, labels = runPersonExtraction(image, threshold=0.99, cpu=True, square=True)

        if patch_images is None:
            continue

        work_canvas = image
        blanc_canvas = np.zeros((360, 480, 3), dtype=np.uint8)
        # blanc_canvas = np.zeros((1280, 720, 3), dtype=np.uint8)
        # goes though all the found human patches
        op = 1
        smaller = "width"
        for (patch, mask, box) in zip(patch_images, masks, boxes):

            # convert patch style
            converted_patch = cg.movie2game(img_path=None, img_tensor=patch)
            # converted_patch = np.transpose(converted_patch, (1, 0, 2))
            actual_height, actual_width = box[1][1] - box[0][1], box[1][0] - box[0][0]

            # resize to the square to the max of both dims
            # this ensures that we have the larger of the sides to the correct size
            resize_dim = max(actual_height, actual_width)
            converted_patch = cv.resize(converted_patch, (resize_dim, resize_dim))

            if resize_dim != actual_width:

                x_start = resize_dim // 2 - actual_width // 2
                x_end = x_start + actual_width

                converted_patch = converted_patch[:, x_start:x_end]
            else:
                smaller = "height"

                y_start = resize_dim // 2 - (actual_height // 2)
                y_end = y_start + actual_height

                converted_patch = converted_patch[y_start:y_end, :]

            # put resized and style transferred patch over background of original image

            # converted_patch = np.transpose(converted_patch, (1, 0, 2))
            # utils.saveIMG(converted_patch, f"cp.jpg", "Results/2_2")

            px1, py1 = box[0]
            px2, py2 = box[1]

            fx1, fy1 = px1, py2 - (px2 - px1)
            fx2, fy2 = px1 + py2 - py1, py2
            height, width, channels = converted_patch.shape

            # if blanc_canvas[py1:py2, px1:px2].shape == converted_patch.shape:
            blanc_canvas[py1:py2, px1:px2] = converted_patch

            # blanc_canvas = np.transpose(blanc_canvas, (1, 0, 2))

            mask = mask.astype(np.uint8)

            D = cv.distanceTransform(mask, cv.DIST_L2, 5)
            D = np.clip(D / 15, 0, 1)
            D = np.stack([D, D, D], axis=2)

            # Broadcast A and B over the distance transform D
            work_canvas = (1 - D) * work_canvas + D * blanc_canvas

        work_canvas = cv.cvtColor(work_canvas, cv.COLOR_RGB2BGR)
        utils.saveIMG(work_canvas, f"conv_frame_{frame_num}.jpg", "Results/2_2")

        converted_frame = work_canvas
        frame_num += 1


def q2_1(model_checkpoint, vid_path, save_path):
    cg = CycleGAN(lim_ds_size=0,
                  BATCH_SIZE=16,
                  checkpoint_path=os.path.join(os.getcwd(), "checkpoints", "q2_1_train"),
                  training_ds="Results/1_1",
                  IMG_WIDTH=IMG_WIDTH,
                  IMG_HEIGHT=IMG_HEIGHT)

    # from q1_1 import runPersonExtraction

    # for now just a few random game frames
    frames = utils.videoImgGenerator("Dataset/Test/Test Movie.mp4", framerate=0)
    frame_num = 0
    for frame in frames:
        original_height, original_width = frame.shape[:2]
        resized_square_frame = cv.resize(frame, (256, 256), interpolation=cv.INTER_LINEAR)
        converted_resized_square_frame = cg.movie2game(img_path=None, img_tensor=resized_square_frame)
        converted_frame = cv.resize(converted_resized_square_frame, (original_width, original_height),
                                    interpolation=cv.INTER_CUBIC)

        cv.imwrite(os.path.join("Frames", "Test", "2_1", f"conv_frame_{frame_num}.jpg"), converted_frame)

        print(frame_num)

        frame_num += 1


def q2_1_train():
    cg = CycleGAN(lim_ds_size=16, BATCH_SIZE=16, IMG_WIDTH=480, IMG_HEIGHT=360,
                  checkpoint_path=os.path.join(os.getcwd(), "checkpoints", "q2_1_train"),
                  training_ds="Frames", FID_save_path="Results/2_1/FID.json")
    cg.mainTrain()


def q2_2_train():
    cg = CycleGAN(lim_ds_size=16, BATCH_SIZE=16,
                  checkpoint_path=os.path.join(os.getcwd(), "checkpoints", "q2_2_train"),
                  training_ds="Results/1_1")
    cg.mainTrain()


if __name__ == '__main__':
    # q2_2()
    q2_1_train()
    # q2_1()
    """cg = CycleGAN(lim_ds_size=16, checkpoint_path=os.path.join(os.getcwd(), "checkpoints", "trai"))

    random_movie_frame_paths = utils.getRandomImges(10, "Results/1_1/Movie")
    random_game_frame_paths = utils.getRandomImges(10, "Results/1_1/Game")

    for g_path in random_game_frame_paths:
        predicted = cg.movie2game(g_path)
        utils.displayImg(predicted)
        utils.saveIMG(predicted, "a.jpg", ".")"""
