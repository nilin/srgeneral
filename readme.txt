git clone --depth 1 https://github.com/nilin/srgeneral

# see installsr.sh for required packages

# make a file:
# localconfig.py
datasetpath='/tmp/'
batch_size = 64

python prox.py --mode ProxSR --dataset mnist --lr 0.1
