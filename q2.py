"""
This code was modified from the CycleGAN tutorial
at: [https://www.tensorflow.org/tutorials/generative/cyclegan?fbclid=IwAR1-7VDGw93g9p7ohYh8MUqNrTaHn_J3etrsQe4AW0MMTXnCc2-loZY77bo]

it also uses the pix2pix model from [https://github.com/tensorflow/examples]
wich has been imported and used in the local pix2pix file in this project
"""

import tensorflow as tf

import pix2pix
import time
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output
import GAN_data_loader
import cv2 as cv

# AUTOTUNE = tf.data.AUTOTUNE

BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3
LAMBDA = 10
EPOCHS = 10

lim_ds_size = 64

"""imgs = [sample_movie[0], to_game[0], sample_game[0], to_movie[0]]
title = ['Horse', 'To Zebra', 'Zebra', 'To Horse']

for img, title in zip(imgs, title):
        cv.imwrite(os.path.join(f"{title}.jpg"), GAN_data_loader.tensor2Image(img))"""


class CycleGAN:
    def __init__(self, BUFFER_SIZE=1000, BATCH_SIZE=1, IMG_WIDTH=256, IMG_HEIGHT=256, OUTPUT_CHANNELS=3, LAMBDA=10,
                 EPOCHS=10, lim_ds_size=64, checkpoint_path="./checkpoints/train"):
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_HEIGHT = IMG_HEIGHT
        self.OUTPUT_CHANNELS = OUTPUT_CHANNELS
        self.LAMBDA = LAMBDA
        self.EPOCHS = EPOCHS
        self.lim_ds_size = lim_ds_size

        self.checkpoint_path = checkpoint_path

        self.game_train_loader, self.game_test_loader, self.movie_train_loader, self.movie_test_loader = GAN_data_loader.DataSetloaders(
            "Results/1_1_edited", batch_size=self.BATCH_SIZE, lim_size=self.lim_ds_size)

        # cv.imwrite(os.path.join(f"movie_train.jpg"), GAN_data_loader.tensor2Image(sample_movie[0]))

        self.generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
        self.generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

        self.discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
        self.discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

        self.loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.ckpt = None
        self.ckpt_manager = None
        self.setupCheckpoints()

    def setupCheckpoints(self):
        self.ckpt = tf.train.Checkpoint(generator_g=self.generator_g,
                                        generator_f=self.generator_f,
                                        discriminator_x=self.discriminator_x,
                                        discriminator_y=self.iscriminator_y,
                                        generator_g_optimizer=self.generator_g_optimizer,
                                        generator_f_optimizer=self.generator_f_optimizer,
                                        discriminator_x_optimizer=self.discriminator_x_optimizer,
                                        discriminator_y_optimizer=self.discriminator_y_optimizer)

        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=5)

    def load(self):
        # if a checkpoint exists, restore the latest checkpoint.
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

    def discriminator_loss(self, real, generated):
        real_loss = self.loss_obj(tf.ones_like(real), real)

        generated_loss = self.loss_obj(tf.zeros_like(generated), generated)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss * 0.5

    def generator_loss(self, generated):
        return self.loss_obj(tf.ones_like(generated), generated)

    def calc_cycle_loss(self, real_image, cycled_image):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

        return LAMBDA * loss1

    def identity_loss(self, real_image, same_image):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return LAMBDA * 0.5 * loss

    def generate_images(self, model, test_input):
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

    @tf.function
    def train_step(self, real_x, real_y):
        # persistent is set to True because the tape is used more than
        # once to calculate the gradients.
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X.

            fake_y = self.generator_g(real_x, training=True)
            cycled_x = self.generator_f(fake_y, training=True)

            fake_x = self.generator_f(real_y, training=True)
            cycled_y = self.generator_g(fake_x, training=True)

            # same_x and same_y are used for identity loss.
            same_x = self.generator_f(real_x, training=True)
            same_y = self.generator_g(real_y, training=True)

            disc_real_x = self.discriminator_x(real_x, training=True)
            disc_real_y = self.discriminator_y(real_y, training=True)

            disc_fake_x = self.discriminator_x(fake_x, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)

            # calculate the loss
            gen_g_loss = self.generator_loss(disc_fake_y)
            gen_f_loss = self.generator_loss(disc_fake_x)

            total_cycle_loss = self.calc_cycle_loss(real_x, cycled_x) + self.calc_cycle_loss(real_y, cycled_y)

            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = gen_g_loss + total_cycle_loss + self.identity_loss(real_y, same_y)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + self.identity_loss(real_x, same_x)

            disc_x_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)

        # Calculate the gradients for generator and discriminator
        generator_g_gradients = tape.gradient(total_gen_g_loss,
                                              self.generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss,
                                              self.generator_f.trainable_variables)

        discriminator_x_gradients = tape.gradient(disc_x_loss,
                                                  self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss,
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

    def mainTrain(self):
        for epoch in range(EPOCHS):
            start = time.time()

            n = 0
            for image_x, image_y in tf.data.Dataset.zip((self.movie_train_loader, self.game_train_loader)):
                train_step(image_x, image_y)
                if n % 10 == 0:
                    print('.', end='')
                n += 1

            clear_output(wait=True)
            # Using a consistent image (sample_horse) so that the progress of the model
            # is clearly visible.
            self.generate_images(self.generator_g, self.sample_horse)

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                    ckpt_save_path))

            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                               time.time() - start))
