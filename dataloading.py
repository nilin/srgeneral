###########################################################################

# based on mnist example from jax authors:
##################################################

# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/jax/blob/main/docs/notebooks/Neural_Network_and_Data_Loading.ipynb) [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/google/jax/blob/main/docs/notebooks/Neural_Network_and_Data_Loading.ipynb)
# 
# **Copyright 2018 The JAX Authors.**
# 
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


###########################################################################
# Data Loading with PyTorch

import jax.numpy as jnp
from jax import random
import numpy as np
from torch.utils import data
from torchvision.datasets import MNIST, CIFAR10
from util import *
from localconfig import *
import os

def getloader(dataset, train):
  n_targets=10

  if dataset=='mnist':
    dummy_imgs = random.normal(random.PRNGKey(1), (batch_size, 28 * 28))
  if dataset=='cifar10':
    dummy_imgs = random.normal(random.PRNGKey(1), (batch_size, 32*32*3))

  dummy_labels = random.normal(random.PRNGKey(1), (batch_size, n_targets))

  def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
      return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
      transposed = zip(*batch)
      return [numpy_collate(samples) for samples in transposed]
    else:
      return np.array(batch)

  class NumpyLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=1,
                  shuffle=False, sampler=None,
                  batch_sampler=None, num_workers=0,
                  pin_memory=False, drop_last=False,
                  timeout=0, worker_init_fn=None):
      super(self.__class__, self).__init__(dataset,
          batch_size=batch_size,
          shuffle=shuffle,
          sampler=sampler,
          batch_sampler=batch_sampler,
          num_workers=num_workers,
          collate_fn=numpy_collate,
          pin_memory=pin_memory,
          drop_last=drop_last,
          timeout=timeout,
          worker_init_fn=worker_init_fn)

  class FlattenAndCast(object):
    def __call__(self, pic):
      return np.ravel(np.array(pic, dtype=jnp.float32))

  if dataset=='mnist':
    thedataset = MNIST(os.path.join(datasetpath,'mnist/'), train=train, download=True, transform=FlattenAndCast())

  if dataset=='cifar10':
    thedataset = CIFAR10(os.path.join(datasetpath,'cifar10/'), train=train, download=True, transform=FlattenAndCast())

  generator = NumpyLoader(thedataset, batch_size=batch_size, num_workers=0, shuffle=True, drop_last=True)

  return generator,(dummy_imgs,dummy_labels)
