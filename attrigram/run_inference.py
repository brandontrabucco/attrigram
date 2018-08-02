# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import numpy as np
import tensorflow as tf

from attrigram import configuration
from attrigram import attrigram_model
import glove

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("input_files", "",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")

tf.flags.DEFINE_integer("num_divisions", 1000,
                       "Number of division to make when running vocabulary.")

tf.logging.set_verbosity(tf.logging.INFO)


def run_attributes(checkpoint_path, filenames, num_divisions):
    g = tf.Graph()
    with g.as_default():
        # Build the model for evaluation.
        model_config = configuration.ModelConfig()
        model = attrigram_model.AttrigramModel(model_config, mode="inference")
        model.build()
        # Create the Saver to restore model Variables.
        saver = tf.train.Saver()
        g.finalize()

    def _restore_fn(sess):
        tf.logging.info("Loading model from checkpoint: %s", checkpoint_path)
        saver.restore(sess, checkpoint_path)
        tf.logging.info("Successfully loaded checkpoint: %s",
                        os.path.basename(checkpoint_path))
        
    # Create the vocabulary.
    vocab = glove.load(model_config.config)[0]
    assert len(vocab.reverse_vocab) % num_divisions == 0, "Vocabulary must be evenly divisible."
    partition_size = len(vocab.reverse_vocab) // num_divisions
    restore_fn = _restore_fn

    with tf.Session(graph=g) as sess:
        # Load the model from checkpoint.
        restore_fn(sess)

        # Prepare the list of image bytes for evaluation.
        topk = []
        for filename in filenames:
            with tf.gfile.GFile(filename, "rb") as f:
                image_bytes = f.read()
            probabilities = []
            for i in range(num_divisions):
                probabilities.extend(sess.run(
                    model.attribute_probabilities, 
                    feed_dict={
                        "image_feed:0": image_bytes,
                        "word_feed:0": [i for i in range(
                            i*partition_size, (i + 1)*partition_size)]}).tolist())
            topk.append(np.argsort(probabilities)[-10:][::-1])

    run_results = []
    for i, k in enumerate(topk):
        run_results.append({"filename": filenames[i], "topk": ""})
        sentence = [vocab.id_to_word(w) for w in k]
        sentence = ", ".join(sentence)
        run_results[i]["topk"] = sentence
                
    return run_results

def main(_):
        
    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        if not checkpoint_path:
            raise ValueError("No checkpoint file found in: %s" % FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path

    filenames = []
    for file_pattern in FLAGS.input_files.split(","):
        filenames.extend(tf.gfile.Glob(file_pattern))
    tf.logging.info("Running caption generation on %d files matching %s",
                  len(filenames), FLAGS.input_files)
    
    attributes = run_attributes(checkpoint_path, filenames, FLAGS.num_divisions)
    for single_attribute in attributes:
        print("Attributes for image %s:" % os.path.basename(single_attribute["filename"]))
        print("    %s " % (single_attribute["topk"]))

if __name__ == "__main__":
    tf.app.run()
