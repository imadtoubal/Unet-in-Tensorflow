# %%
from utils import *
import numpy as np
from numpy.random import seed
# from tensorflow import set_random_seed
import tensorflow as tf
from datetime import datetime
from losses import fl

# Custom imports
from nets import scSEunetpp
from config import get_config

#seed(0)
#set_random_seed(0)

# %% Import configs
cfg = get_config()


# %% load data
X,  Y  = load_dataset(cfg['data']['train_path'])
Xv, Yv = load_dataset(cfg['data']['val_path'])

# %% train
model = scseUnetpp(128, 128, 64, 1)
model_name = '/model_scseunetpp_{}.p5'.format(datetime.now().strftime('%m%d%Y%H%M%S'))
checkpointer = tf.keras.callbacks.ModelCheckpoint(cfg['train']['ckpt']['ckpt_path'] + model_name, **cfg['train']['ckpt'])

callbacks = [
            checkpointer,
            tf.keras.callbacks.EarlyStopping(**cfg['train']['early_stopping']),
            tf.keras.callbacks.TensorBoard(log_dir='logs/fit/{}'.format(model_name), profile_batch=0)
            ]

Yt  = {'out_{}'.format(o):Y  for o in range(len(model.outputs))}
Ytv = {'out_{}'.format(o):Yv for o in range(len(model.outputs))}

model.fit(X, Yt,batch_size=1, validation_data=(Xv, Ytv), callbacks=callbacks, **cfg['net'])