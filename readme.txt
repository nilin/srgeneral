git clone --depth 1 https://github.com/nilin/srgeneral

# python 3.10 environment

pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install torch==1.11.0 torchvision
pip install optax flax kfac-jax

python prox.py
