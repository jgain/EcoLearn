from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time
import sys
import PIL.Image
import zmq
import subprocess
from scipy import signal
from numpngw import write_png

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--output_dir", required=False, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")
parser.add_argument("--checkpoint2", default=None, help="directory with second checkpoint to resume training from or use for testing")

parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--separable_conv", action="store_true", help="use separable convolutions in the generator")
parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=286, help="scale images to this size before cropping to crop_size")
parser.add_argument("--crop_size", type=int, default=256, help="crop images to this size before applying to the generator")
parser.add_argument("--png16bits", dest="png16bits", action="store_true", help="use png 16 bits images encoder and decoders")
parser.set_defaults(png16bits=False)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")

# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
a = parser.parse_args()

EPS = 1e-12
CROP_SIZE = a.crop_size

Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")


def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2

def discrim_conv(batch_input, out_channels, stride):
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))


def gen_conv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if a.separable_conv:
        return tf.layers.separable_conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)


def gen_deconv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if a.separable_conv:
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))


def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image

# based on https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c
def rgb_to_lab(srgb):
    with tf.name_scope("rgb_to_lab"):
        srgb = check_image(srgb)
        srgb_pixels = tf.reshape(srgb, [-1, 3])

        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334], # R
                [0.357580, 0.715160, 0.119193], # G
                [0.180423, 0.072169, 0.950227], # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("xyz_to_cielab"):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])

            epsilon = 6/29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [  0.0,  500.0,    0.0], # fx
                [116.0, -500.0,  200.0], # fy
                [  0.0,    0.0, -200.0], # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(srgb))


def lab_to_rgb(lab):
    with tf.name_scope("lab_to_rgb"):
        lab = check_image(lab)
        lab_pixels = tf.reshape(lab, [-1, 3])

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("cielab_to_xyz"):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1/116.0, 1/116.0,  1/116.0], # l
                [1/500.0,     0.0,      0.0], # a
                [    0.0,     0.0, -1/200.0], # b
            ])
            fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

            # convert to xyz
            epsilon = 6/29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask

            # denormalize for D65 white point
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope("xyz_to_srgb"):
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [ 3.2404542, -0.9692660,  0.0556434], # x
                [-1.5371385,  1.8760108, -0.2040259], # y
                [-0.4985314,  0.0415560,  1.0572252], # z
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            # avoid a slightly negative number messing up the conversion
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask

        return tf.reshape(srgb_pixels, tf.shape(lab))

def create_generator(generator_inputs, generator_outputs_channels):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = gen_conv(generator_inputs, a.ngf)
        layers.append(output)

    layer_specs = [
        a.ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        a.ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        a.ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        a.ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        a.ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        a.ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        a.ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (a.ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (a.ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (a.ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (a.ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (a.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (a.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (a.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = gen_deconv(rectified, out_channels)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = gen_deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]


class cganmodel:
    def __init__(self,checkpointname,sess=None, gpu_frac=0.4):
        self.gpu_frac = gpu_frac
        def create_discriminator(discrim_inputs, discrim_targets):
            n_layers = 3
            layers = []

            # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
            input = tf.concat([discrim_inputs, discrim_targets], axis=3)

            # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
            with tf.variable_scope("layer_1"):
                convolved = discrim_conv(input, a.ndf, stride=2)
                rectified = lrelu(convolved, 0.2)
                layers.append(rectified)

            # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
            # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
            # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
            for i in range(n_layers):
                with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                    out_channels = a.ndf * min(2**(i+1), 8)
                    stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                    convolved = discrim_conv(layers[-1], out_channels, stride=stride)
                    normalized = batchnorm(convolved)
                    rectified = lrelu(normalized, 0.2)
                    layers.append(rectified)

            # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                convolved = discrim_conv(rectified, out_channels=1, stride=1)
                output = tf.sigmoid(convolved)
                layers.append(output)

            return layers[-1]


        if sess is not None:
            self.sess_own = False
            self.sess = sess
        else:
            self.sess_own = True
            self.sess = None
            self.graph = tf.Graph()
            
        with self.graph.as_default():

            inputs = tf.placeholder(tf.float32, shape=(1,CROP_SIZE,CROP_SIZE,3))
            targets = tf.placeholder(tf.float32, shape=(1,CROP_SIZE,CROP_SIZE,3))

            with tf.variable_scope("generator"):
                out_channels = int(targets.get_shape()[-1])
                outputs = create_generator(inputs, out_channels)

            # create two copies of discriminator, one for real pairs and one for fake pairs
            # they share the same underlying variables
            with tf.name_scope("real_discriminator"):
                with tf.variable_scope("discriminator"):
                    # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
                    predict_real = create_discriminator(inputs, targets)

            with tf.name_scope("fake_discriminator"):
                with tf.variable_scope("discriminator", reuse=True):
                    # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
                    predict_fake = create_discriminator(inputs, outputs)

            with tf.name_scope("discriminator_loss"):
                # minimizing -tf.log will try to get inputs to 1
                # predict_real => 1
                # predict_fake => 0
                discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

            with tf.name_scope("generator_loss"):
                # predict_fake => 1
                # abs(targets - outputs) => 0
                gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
                gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
                gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight

            with tf.name_scope("discriminator_train"):
                discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
                discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
                discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
                discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

            with tf.name_scope("generator_train"):
                with tf.control_dependencies([discrim_train]):
                    gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
                    gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
                    gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
                    gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

            ema = tf.train.ExponentialMovingAverage(decay=0.99)
            update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

            global_step = tf.train.get_or_create_global_step()
            incr_global_step = tf.assign(global_step, global_step+1)

        self.predict_real=predict_real,
        self.predict_fake=predict_fake,
        self.discrim_loss=ema.average(discrim_loss),
        self.discrim_grads_and_vars=discrim_grads_and_vars,
        self.gen_loss_GAN=ema.average(gen_loss_GAN),
        self.gen_loss_L1=ema.average(gen_loss_L1),
        self.gen_grads_and_vars=gen_grads_and_vars,
        self.outputs=outputs,
        self.train=tf.group(update_losses, incr_global_step, gen_train)
        self.input_tensor = inputs
        self.checkpointname = checkpointname

    def init_vars(self):
        with self.graph.as_default():
            saver = tf.train.Saver(max_to_keep=1)
            #checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            checkpoint = tf.train.latest_checkpoint(self.checkpointname)
            saver.restore(self.sess, checkpoint)
            #self.loader = self.loader.restore(self.sess, os.path.splitext(self.model_filename)[0])
            #self.graph = tf.get_default_graph()

            self.output_tensor = self.graph.get_tensor_by_name("generator/decoder_1/Tanh:0")

            self.metadata = tf.RunMetadata()

    def __enter__(self):
        self.open(self.gpu_frac)
        return self
    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        if self.sess_own:
            self.sess.close()

    def open(self, gpu_frac):
        if self.sess is None:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
            self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(gpu_options=gpu_options))
            #self.sess = tf.Session()
        self.init_vars()

def save_images(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets


def append_index(filesets, step=False):
    index_path = os.path.join(a.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["inputs", "outputs", "targets"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path


def main():
    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    #if not os.path.exists(a.output_dir):
    #    os.makedirs(a.output_dir)

    if a.checkpoint is None:
        raise Exception("checkpoint required for test mode")

    # load some options from the checkpoint
    options = {"which_direction", "ngf", "ndf", "lab_colorization"}
    with open(os.path.join(a.checkpoint, "options.json")) as f:
        for key, val in json.loads(f.read()).items():
            if key in options:
                print("loaded", key, "=", val)
                setattr(a, key, val)
    # disable these features in test mode
    #a.scale_size = CROP_SIZE # TODO cleaner

    for k, v in a._get_kwargs():
        print(k, "=", v)

    #with open(os.path.join(a.output_dir, "options.json"), "w") as f:
    #    f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    input2ph = tf.placeholder(tf.float32, shape=(1,CROP_SIZE,CROP_SIZE,3))
    output2ph = tf.placeholder(tf.float32, shape=(1,CROP_SIZE,CROP_SIZE,3))

    # inputs and targets are [batch_size, height, width, channels]

    model = cganmodel(a.checkpoint)
    model.open(0.25)
    model2 = cganmodel(a.checkpoint2)
    model2.open(0.25)

    def convert(image):
        if a.aspect_ratio != 1.0:
            # upscale to correct aspect ratio
            size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        if a.png16bits:
            return tf.image.convert_image_dtype(image, dtype=tf.uint16, saturate=True)
        else:
            return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    
    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    def prepare_image(image):
        image = image.reshape([CROP_SIZE,CROP_SIZE,3])

        #inputfile = "input.png"
        #image = PIL.Image.open(inputfile)   
        #image = np.array(image)
        #image = image.astype(np.uint16)
        #image = image.reshape([CROP_SIZE,CROP_SIZE,3])

        image = image.astype(np.float32)
        image = (np.array(image)/65535.0*2)-1.0;# truc louche ici a checker -> attention mettre 65535 quand ce sera ok
        colorimage = image
        inputimg = np.reshape(colorimage[:,0:CROP_SIZE,:],(1,CROP_SIZE,CROP_SIZE,-1))
        return inputimg

    def get_image_from_socket(socket):
        try:
            #print("Checking if image received...", end="")
            message = socket.recv(flags=zmq.NOBLOCK,copy=True,track=False)
        except zmq.Again as e:
            #print("none")
            time.sleep(0.1)
            return None, None, None

        #print("Image received")
        start = time.time()
        sys.stdout.flush()
        buf = memoryview(message)
        image = np.frombuffer(buf, dtype=np.uint16)
        print("Image received. Size: {}".format(image.shape))
        return image, buf, start

    def create_socket_and_context():
        context = zmq.Context()
        socket = context.socket(zmq.REP) 
        socket.bind("tcp://*:5555")
        print("Waiting for an incoming image...")
        sys.stdout.flush()
        return socket, context

    def deprocess_outputs(result):
        deprocessed_outputs = deprocess(result)
        out = np.reshape(deprocessed_outputs,(CROP_SIZE,CROP_SIZE,3))
        out = out*65535.0;
        out = out.reshape([CROP_SIZE*CROP_SIZE*3])
        out = out.astype(np.uint16)
        return out

    def get_debug_image(inputfile=None):
        import png
        if inputfile is None:
            inputfile = "input.png"
        #image = PIL.Image.open(inputfile)   
        #image = np.array(image)
        w, h, data, info = png.Reader(inputfile).read()
        image = np.array(list(data))
        #image = image.astype(np.uint16)
        image = image.reshape([CROP_SIZE*CROP_SIZE*3])
        return image

    def write_debug_image(arr, outfilename=None):
        import png
        arr = arr.reshape([CROP_SIZE, CROP_SIZE * 3])
        if outfilename is None:
            outfilename = "output_test.png"
        writer = png.Writer(arr.shape[1] // 3, arr.shape[0], bitdepth=16, greyscale=False, interlace=0, alpha=False, planes=3)
        with open(outfilename, "wb+") as outfile:
            writer.write(outfile, arr)

    def fix_result(result):
        chm_part = result[0,:,:,1]
        chm_part[chm_part < -0.5] = -1
        chm_part[chm_part > -1] = 127 * 256 / 65535.0 * 2.0 - 1

    def show_channels(arr, chlist=None):
        import matplotlib.pyplot as plt
        if chlist is None:
            chlist = [0, 1, 2]
        for i in chlist:
            plt.imshow(arr[0,:,:,i])
            plt.show()


    socket, context = create_socket_and_context()
    # demon mode

    interface_proc = subprocess.Popen(["./viewer", str(a.scale_size)])

    while True: 
        image, buf, start = get_image_from_socket(socket)       # returning the buf object, since I do not know if it will get cleaned up if its scope is only the function
        if image is None:
            if interface_proc.poll() is not None:
                return 0
            continue
        inputimg = prepare_image(image) 
        # exec
        nnoutput = model.output_tensor
        result = model.sess.run(nnoutput, feed_dict={model.input_tensor: inputimg})

        fix_result(result)
        result[0,:,:,0] = inputimg[0,:,:,0]

        nnoutput2 = model2.output_tensor
        result2 = model2.sess.run(nnoutput2, feed_dict={model2.input_tensor: result})

        # deprocess and write image in 16 bits mode
        out = deprocess_outputs(result2)

        end = time.time()
        print("Elapsed time to evaluate the NN : %f" % (end-start))
        sys.stdout.flush()

        #inputfile = "input.png"
        #image = PIL.Image.open(inputfile)   
        #image = np.array(image)
        #image = image.astype(np.uint16)
        #image = image.reshape([CROP_SIZE*CROP_SIZE*3])

        socket.send(out,flags=0,copy=True,track=False)
        #write_debug_image(out, "~/gpu_gui_out.png")


main()
