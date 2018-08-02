# Copyright 2016 Brandon Trabucco. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Image-to-attribute-grams implementation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
import os.path
import glove.utils

from attrigram.ops import image_embedding
from attrigram.ops import image_processing
from attrigram.ops import inputs as input_ops


class AttrigramModel(object):
    """Image-to-attribute-grams implementation.
    """

    def __init__(self, config, mode, train_inception=False):
        """Basic setup.

        Args:
            config: Object containing configuration parameters.
            mode: "train", "eval" or "inference".
            train_inception: Whether the inception submodel variables are trainable.
        """
        assert mode in ["train", "eval", "inference"]
        self.config = config
        self.mode = mode
        self.train_inception = train_inception

        # Reader for the input data.
        self.reader = tf.TFRecordReader()

        # To match the "Show and Tell" paper we initialize all variables with a
        # random uniform initializer.
        self.initializer = tf.random_uniform_initializer(
                minval=-self.config.initializer_scale,
                maxval=self.config.initializer_scale)

    def is_training(self):
        """Returns true if the model is built for training mode."""
        return self.mode == "train"

    def process_image(self, encoded_image, thread_id=0):
        """Decodes and processes an image string.

        Args:
            encoded_image: A scalar string Tensor; the encoded image.
            thread_id: Preprocessing thread id used to select the ordering of color
                distortions.

        Returns:
            A float32 Tensor of shape [height, width, 3]; the processed image.
        """
        return image_processing.process_image(encoded_image,
                                              is_training=self.is_training(),
                                              height=self.config.image_height,
                                              width=self.config.image_width,
                                              thread_id=thread_id,
                                              image_format=self.config.image_format)
    
    def build_inference_inputs(self):
        """Inputs for running inference on the model.
        
        Outputs:
            self.inference_images (inference only)
            self.inference_words (inference only)
        """
        if self.mode == "inference":
            inference_images = tf.placeholder(dtype=tf.string, 
                                        shape=[], 
                                        name="image_feed")
            inference_images = self.process_image(inference_images)
            inference_images = tf.expand_dims(inference_images, axis=0)
            inference_words = tf.placeholder(dtype=tf.int32, 
                                       shape=[None], 
                                       name="word_feed")
            inference_words = tf.expand_dims(inference_words, axis=0)
        else:
            inference_images = None
            inference_words = None
            
        self.inference_images = inference_images
        self.inference_words = inference_words

    def build_deepfashion_inputs(self):
        """Input prefetching, preprocessing and batching.

        Outputs:
            self.deepfashion_filenames (training and eval only)
            self.deepfashion_images (training and eval only)
            self.deepfashion_categories (training and eval only)
            self.deepfashion_attributes (training and eval only)
            self.deepfashion_words (training and eval only)
        """
        if self.mode == "inference":
            # No target sequences or input mask in inference mode.
            deepfashion_filenames = None
            deepfashion_images = None
            deepfashion_categories = None
            deepfashion_attributes = None
            deepfashion_words = None

        else:
            # Prefetch serialized SequenceExample protos.
            input_queue = input_ops.prefetch_input_data(
                self.reader,
                self.config.input_file_pattern,
                is_training=self.is_training(),
                batch_size=self.config.batch_size,
                values_per_shard=self.config.values_per_input_shard,
                input_queue_capacity_factor=self.config.input_queue_capacity_factor,
                num_reader_threads=self.config.num_input_reader_threads)

            # Image processing and random distortion. Split across multiple threads
            # with each thread applying a slightly different distortion.
            assert self.config.num_preprocess_threads % 2 == 0
            images_and_annotations = []
            for thread_id in range(self.config.num_preprocess_threads):
                serialized_sequence_example = input_queue.dequeue()
                filename, encoded_image, category, attributes, labels = (
                    input_ops.parse_sequence_example(serialized_sequence_example))
                image = self.process_image(encoded_image, thread_id=thread_id)
                images_and_annotations.append([filename, image, category, attributes, labels])

            # Batch inputs.
            queue_capacity = (2 * self.config.num_preprocess_threads *
                                                self.config.batch_size)
            (deepfashion_filenames, 
                deepfashion_images, 
                deepfashion_categories, 
                deepfashion_words,
                deepfashion_attributes) = (
                    input_ops.batch_with_dynamic_pad(images_and_annotations,
                                                     vocab_size=self.config.vocab_size,
                                                     batch_size=self.config.batch_size,
                                                     queue_capacity=queue_capacity))
            deepfashion_attributes = tf.cast(deepfashion_attributes, tf.float32)
            
        self.deepfashion_filenames = deepfashion_filenames
        self.deepfashion_images = deepfashion_images
        self.deepfashion_categories = deepfashion_categories
        self.deepfashion_words = deepfashion_words
        self.deepfashion_attributes = deepfashion_attributes
        
    def build_word_embeddings(self):
        """Builds the word embedding subgraph and glove vectors.
        
        Outputs:
            self.word_embedding_map
            self.inference_word_embeddings
            self.deepfashion_word_embeddings
        """
        with tf.variable_scope("word_embedding"):

            word_embedding_map = tf.get_variable(
                name="word_embedding_map",
                initializer=tf.constant(glove.load(self.config.config)[1], dtype=tf.float32),
                trainable=self.config.train_embeddings)
            
            if self.mode == "inference":
                inference_word_embeddings = tf.nn.embedding_lookup(
                    word_embedding_map, self.inference_words)
                deepfashion_word_embeddings = None
                
            else:
                inference_word_embeddings = None
                deepfashion_word_embeddings = tf.nn.embedding_lookup(
                    word_embedding_map, self.deepfashion_words)
                
        self.word_embedding_map = word_embedding_map
        self.inference_word_embeddings = inference_word_embeddings
        self.deepfashion_word_embeddings = deepfashion_word_embeddings

    def build_image_embeddings(self):
        """Builds the image model subgraph and generates attributes.

        Inputs:
            self.images

        Outputs:
            self.image_embeddings
        """
        if self.mode == "inference":
            word_embeddings = self.inference_word_embeddings
            images = self.inference_images
            
        else:
            word_embeddings = self.deepfashion_word_embeddings
            images = self.deepfashion_images
            
        # Compute the image features.
        inception_output = image_embedding.inception_v3(
            images,
            trainable=self.train_inception,
            is_training=self.is_training())
        self.inception_variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")
        
        # Compute the embedding of the image features.
        image_embedding_map = tf.layers.Dense(
            units=self.config.embedding_size, 
            use_bias=False, kernel_initializer=self.initializer)
        image_embeddings = image_embedding_map(inception_output)
        image_embeddings = tf.expand_dims(image_embeddings, 1)
        
        # Compute the normalized sigmoid attention across the image.
        image_attention_layer = tf.layers.Dense(
            units=1, activation=tf.sigmoid, kernel_initializer=self.initializer)
        
        # Compute the probability of word present.
        logits_layer = tf.layers.Dense(
            units=1, kernel_initializer=self.initializer)
        
        word_embeddings = tf.expand_dims(
            word_embeddings, axis=2)
        word_embeddings = tf.expand_dims(
            word_embeddings, axis=3)
        word_embeddings = tf.tile(word_embeddings, [1, 1, 
            tf.shape(image_embeddings)[2], 
            tf.shape(image_embeddings)[3], 1])

        # Dual embedding shape: [batch_size, word_size, x, y, embedding_size]
        word_image = image_embeddings * word_embeddings
        image_attention = image_attention_layer(word_image)
        image_attention = image_attention / tf.cast(
            tf.shape(word_image)[2] * tf.shape(word_image)[3], 
            dtype=tf.float32)
        image_context = tf.reduce_sum(
            image_embeddings * image_attention, 
            axis=[2, 3])

        # Word probability shape: [batch_size, word_size, 1]
        attribute_logits = tf.squeeze(logits_layer(image_context))
        attribute_probabilities = tf.sigmoid(attribute_logits)
       
        self.inception_output = inception_output
        self.image_embedding = image_embedding
        self.attribute_logits = attribute_logits
        self.attribute_probabilities = attribute_probabilities

    def build_losses(self):
        """Builds the losses on which to optimize the model.
        Inputs:
            self.attribute_logits
            self.attribute_probabilities

        Outputs:
            self.total_loss
        """
        if self.mode != "inference":

            # Compute losses.
            deepfashion_losses = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.deepfashion_attributes, logits=self.attribute_logits)
            deepfashion_loss = tf.reduce_mean(deepfashion_losses)
            tf.losses.add_loss(deepfashion_loss)

            total_loss = tf.losses.get_total_loss()

            # Add summaries.
            tf.summary.scalar("losses/deepfashion_loss", deepfashion_loss)
            tf.summary.scalar("losses/total_loss", total_loss)
            for var in tf.trainable_variables():
                tf.summary.histogram("parameters/" + var.op.name, var)

            self.total_loss = total_loss

    def setup_inception_initializer(self):
        """Sets up the function to restore inception variables from checkpoint."""
        if self.mode != "inference":
            # Restore inception variables only.
            saver = tf.train.Saver(self.inception_variables)

            def restore_fn(sess):
                tf.logging.info("Restoring Inception variables from checkpoint file %s",
                    self.config.inception_checkpoint_file)
                saver.restore(sess, self.config.inception_checkpoint_file)

            self.init_fn = restore_fn

    def setup_global_step(self):
        """Sets up the global step Tensor."""
        global_step = tf.Variable(
            initial_value=0, name="global_step", trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        self.global_step = global_step

    def build(self):
        """Creates all ops for training and evaluation."""
        self.build_inference_inputs()
        self.build_deepfashion_inputs()
        self.build_word_embeddings()
        self.build_image_embeddings()
        self.build_losses()
        self.setup_inception_initializer()
        self.setup_global_step()

