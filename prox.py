import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax
from jax.scipy.special import logsumexp
import numpy as np
from torch.utils import data
from torchvision.datasets import MNIST
import optax
import time
import flax
import flax.linen as nn
from jax import random as rnd
from util import *
from jax.tree_util import tree_flatten, tree_unflatten, tree_map


rate = 0.1
num_epochs = 8
batch_size = 32
n_targets = 10


class Model(nn.Module):
  @nn.compact
  def __call__(self, x, probs=False, raw=False):
    x = jnp.reshape(x,x.shape[:-1]+(28,28,1))
    x = nn.Conv(features=8, strides=2, kernel_size=(3, 3))(x)
    x = nn.LayerNorm()(x)
    x = nn.relu(x)
    x = nn.Conv(features=16, strides=2, kernel_size=(3, 3))(x)
    x = nn.LayerNorm()(x)
    x = nn.relu(x)
    x = jnp.reshape(x,x.shape[:-3]+(-1,))
    x = nn.Dense(features=512)(x)
    x = nn.LayerNorm()(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    return nn.log_softmax(x)

model=Model()

batched_predict = model.apply
dummy_imgs = random.normal(random.PRNGKey(1), (batch_size, 28 * 28))
dummy_labels = random.normal(random.PRNGKey(1), (batch_size, n_targets))
params=model.init(rnd.PRNGKey(0), dummy_imgs)

def batched_predict_restricted(params,images,targets):
  prediction=batched_predict(params,images)
  return jnp.sum(prediction*targets,axis=-1)

jac_restricted=jax.jacrev(batched_predict_restricted)

def one_hot(x, k, dtype=jnp.float32):
  return jnp.array(x[:, None] == jnp.arange(k), dtype)

def accuracy(params, images, targets):
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
  return jnp.mean(predicted_class == target_class)



#########################
### method with Gil's momentum scheme ###

@jax.jit
def newgradmomentum(params,images,targets,prev_grad):

  O=jac_restricted(params,images,targets)

  def flattenjac(O):
    O=jax.tree_map(lambda A:jnp.reshape(A,(A.shape[0],-1)),O)
    O,_=jax.tree_flatten(O)
    O=jnp.concatenate(O,axis=-1)
    return O

  Ohat=O_=flattenjac(O)
  T=jnp.inner(O_,O_)

  l,v=jnp.linalg.eigh(T)
  valid=l>jnp.quantile(l,.5)
  inv=(1/l)*valid
  invT=v*inv[None,:] @ v.T

  logits=jnp.sum(batched_predict(params,images)*targets,axis=-1)
  p=jnp.exp(logits)
  e=-(1-p)


# Gil's momentum scheme

  prev_grad,_=tree_flatten(prev_grad)
  prev_grad=jnp.concatenate([jnp.ravel(A) for A in prev_grad])

  Ohat_prev_grad = Ohat @ prev_grad
  OhatT_Tinv = Ohat.T @ invT
  
  #

  min_sr_solution = Ohat.T @ (invT @ e)
  prev_grad_subspace = OhatT_Tinv @ Ohat_prev_grad
  prev_grad_complement = prev_grad - prev_grad_subspace
#
  prev_grad_parallel = (
      min_sr_solution
      * (min_sr_solution @ prev_grad_subspace)
      / (min_sr_solution @ min_sr_solution)
  )

  out=(
    0.1*min_sr_solution
    +.9*prev_grad_parallel
    +.99*prev_grad_complement
  )

  treelist,treeshape=tree_flatten(O)
  sizes=[A.size for A in treelist]
  shapes=[A.shape for A in treelist]

  out_=[]
  start=0
  for S,sh in zip(sizes,shapes):
    s=S//batch_size
    block=out[start:start+s]
    out_.append(jnp.reshape(block,sh[1:]))
    start=start+s
  
  return tree_unflatten(treeshape,out_),\
  dict(loss=-jnp.mean(logits)/n_targets)

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

# Define our dataset, using torch datasets
mnist_dataset = MNIST('/tmp/mnist/', download=True, transform=FlattenAndCast())
training_generator = NumpyLoader(mnist_dataset, batch_size=batch_size, num_workers=0)

# Get the full train dataset (for checking accuracy while training)
train_images = np.array(mnist_dataset.train_data).reshape(len(mnist_dataset.train_data), -1)
train_labels = one_hot(np.array(mnist_dataset.train_labels), n_targets)

# Get full test dataset
mnist_dataset_test = MNIST('/tmp/mnist/', download=True, train=False)
test_images = jnp.array(mnist_dataset_test.test_data.numpy().reshape(len(mnist_dataset_test.test_data), -1), dtype=jnp.float32)
test_labels = one_hot(np.array(mnist_dataset_test.test_labels), n_targets)

# End data loading
###########################################################################

###########################################################################
# Training Loop


import matplotlib.pyplot as plt
import pickle

mode=input('mode (kfac/new): ')

if mode=='new' or mode=='newmomentum':
  @jax.jit
  def update(params,grads,rate):
    return tree_map(lambda p,g:p-rate*g,params,grads)
  
if mode=='kfac':
  import kfac_jax

  def loss(params, xy):
    x,y=xy
    logits = batched_predict(params, x)
    kfac_jax.register_softmax_cross_entropy_loss(logits, y)
    return -jnp.mean(logits * y)

  kfac_opt=kfac_jax.Optimizer(
    value_and_grad_func=jax.value_and_grad(loss),
    l2_reg=.001,
    value_func_has_aux=False,
    value_func_has_state=False,
    value_func_has_rng=False,
    use_adaptive_learning_rate=True,
    use_adaptive_momentum=True,
    use_adaptive_damping=True,
    initial_damping=1.0,
    multi_device=False,
  )
  optstate=kfac_opt.init(params,rnd.PRNGKey(0),(dummy_imgs,dummy_labels))
  key=rnd.PRNGKey(0)


losses=[]
accuracies=[]

prevgrad=tree_map(lambda x:0*x,params)

for epoch in range(num_epochs):
  start_time = time.time()
  for i, (x, y) in enumerate(training_generator):
    y = one_hot(y, n_targets)

    if mode=='new':
      grads, aux = newgradmomentum(params, x, y ,prevgrad)
      prevgrad=grads
      params = update(params, grads, rate)

    if mode=='kfac':
      key=rnd.split(key)[0]
      params,optstate,aux=kfac_opt.step(params,optstate,key,batch=(x,y),global_step_int=i)

    losses.append(aux['loss'])
    accuracies.append(accuracy(params, x, y))

    print(losses[-1])
    if i%100==0 and i>=100:

      with open('outputs/{}.pkl'.format(mode),'wb') as f:
        pickle.dump((losses,accuracies),f)

    if i%100==0:
      fig,axs=plt.subplots(2)

      for mode_ in ['new','kfac']:
        try:
          with open('outputs/{}.pkl'.format(mode_),'rb') as f:
            losses_,accuracies_=pickle.load(f)

          modelabel=mode_
          axs[0].plot(losses_,label=modelabel)
          axs[1].plot(accuracies_,label=modelabel)
          axs[0].legend()
          axs[1].legend()
        except:
          print('no data for {}'.format(mode_))

      plt.savefig('outputs/loss.png')
      plt.close()

  epoch_time = time.time() - start_time

  train_acc = accuracy(params, train_images, train_labels)
  test_acc = accuracy(params, test_images, test_labels)
  print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
  print("Training set accuracy {}".format(train_acc))
  print("Test set accuracy {}".format(test_acc))