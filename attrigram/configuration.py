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

"""Image-to-text model and training configurations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import glove.configuration


class ModelConfig(object):
  """Wrapper class for model hyperparameters."""

  def __init__(self):
    """Sets the default model hyperparameters."""
    # File pattern of sharded TFRecord file containing SequenceExample protos.
    # Must be provided in training and evaluation modes.
    self.input_file_pattern = None

    # Image format ("jpeg" or "png").
    self.image_format = "jpeg"

    # Approximate number of values per input shard. Used to ensure sufficient
    # mixing between shards in training.
    self.values_per_input_shard = 2300
    # Minimum number of shards to keep in the input queue.
    self.input_queue_capacity_factor = 2
    # Number of threads for prefetching SequenceExample protos.
    self.num_input_reader_threads = 1

    # Number of unique words in the vocab (plus 1, for <UNK>).
    # The default value is larger than the expected actual vocab size to allow
    # for differences between tokenizer versions used in preprocessing. There is
    # no harm in using a value greater than the actual vocab size, but using a
    # value less than the actual vocab size will result in an error.
    self.vocab_size = 70000

    # Number of threads for image preprocessing. Should be a multiple of 2.
    self.num_preprocess_threads = 4

    # Batch size.
    self.batch_size = 32
    self.beam_size = 3
    self.maximum_iterations = 20
    
    # File containing an Inception v3 checkpoint to initialize the variables
    # of the Inception model. Must be provided when starting training for the
    # first time.
    self.inception_checkpoint_file = None

    # Dimensions of Inception v3 input images.
    self.image_height = 299
    self.image_width = 299

    # Scale used to initialize model variables.
    self.initializer_scale = 0.08
    self.embedding_size = 300

    # Should we fine tune the glove embeddings, etc.
    self.train_embeddings = False

    # Deep fashion dataset loading parameters.
    self.deepfashion_file_pattern = None
    self.df_filename_name = "image/filename"
    self.df_image_name = "image/data"
    self.df_category_name = "image/category"
    self.df_attributes_name = "image/attributes"

    # The config params for loading the vocab and embedding
    self.config = glove.configuration.Configuration(
        embedding=300,
        filedir="/home/ubuntu/research/data/glove/embeddings/",
        length=70000,
        start_word="<S>",
        end_word="</S>",
        unk_word="<UNK>")

   

class TrainingConfig(object):
  """Wrapper class for training hyperparameters."""

  def __init__(self):
    """Sets the default training hyperparameters."""
    # Number of examples per epoch of training data.
    self.num_examples_per_epoch = 586363

    # Optimizer for training the model.
    self.optimizer = "SGD"

    # Learning rate for the initial phase of training.
    self.initial_learning_rate = 2.0
    self.learning_rate_decay_factor = 0.5
    self.num_epochs_per_decay = 8.0

    # Learning rate when fine tuning the Inception v3 parameters.
    self.train_inception_learning_rate = 0.0005

    # If not None, clip gradients to this value.
    self.clip_gradients = 5.0

    # How many model checkpoints to keep.
    self.max_checkpoints_to_keep = 5
