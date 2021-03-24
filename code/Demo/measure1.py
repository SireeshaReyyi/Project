##########################################################################
'''
                       U-Net Training Code
code to train U-Net Model for the automatic detection of coronary artery from cardiac medical images"
'''
############################################################################

import numpy as np
from data_dlb_loader import load_train_data, load_test_data,load_validation_data
from data_dlb_loader import prep_test_data
from unet_model import dlb_Unet
from keras import models
from keras.callbacks import TensorBoard,EarlyStopping,ReduceLROnPlateau
from keras.optimizers import Adam,SGD
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN, device=gpu0, floatX=float32, optimizer=fast_compile'
#############################################################################

# Model Invocation
model = dlb_Unet((512,512,1))
#tunining Learning Rate
#rlp=ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', min_lr=1e-5)
sgd = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)

model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
print ('Compiled: OK')

# Model visualization
#from keras.utils.vis_utils import plot_model
#plot_model(model, to_file='model.png', show_shapes=True)


# load  and normalization of trian and validation data

imgs_train, imgs_mask_train = load_train_data()
imgs_train = imgs_train.astype('float32')
mean = np.mean(imgs_train)  
std = np.std(imgs_train)  
imgs_train -= mean
imgs_train /= std
imgs_mask_train = imgs_mask_train.astype('float32')
imgs_mask_train /= 255. 

imgs_valid, imgs_mask_valid = load_validation_data()
imgs_valid = imgs_valid.astype('float32')
mean = np.mean(imgs_valid) 
std = np.std(imgs_valid)  
imgs_valid -= mean
imgs_valid /= std

imgs_mask_valid = imgs_mask_valid.astype('float32')
imgs_mask_valid /= 255. 

# Train Model 

nb_epoch = 70
batch_size = 4
#es=EarlyStopping(monitor='val_loss', min_delta=0.5, patience=2, verbose=0, mode='auto')
tb=TensorBoard(log_dir='dlb_logs_70ep')
cb_list = [tb]


import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
class Metrics(Callback):
def on_train_begin(self, logs={}):
 self.val_f1s = []
 self.val_recalls = []
 self.val_precisions = []
 
def on_epoch_end(self, epoch, logs={}):
 val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
 val_targ = self.model.validation_data[1]
 _val_f1 = f1_score(val_targ, val_predict)
 _val_recall = recall_score(val_targ, val_predict)
 _val_precision = precision_score(val_targ, val_predict)
 self.val_f1s.append(_val_f1)
 self.val_recalls.append(_val_recall)
 self.val_precisions.append(_val_precision)
 print “ — val_f1: %f — val_precision: %f — val_recall %f” %(_val_f1, _val_precision, _val_recall)
 return
 
metrics = Metrics()
print (metrics.val_f1s)
model.fit(imgs_train, imgs_mask_train, 
 validation_data=(imgs_valid , imgs_mask_valid),
 nb_epoch=10,
 batch_size=64,
 callbacks=[metrics])
