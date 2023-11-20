import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # No display of info, 0,1,2,3

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import Input, Model
from scipy import sparse

#-----------------------------------parameters-------------------------------------------------------#

n, code_size, method = 6, 56, 'mlp'
p_i, p_d, p_im, p_pfd = 0.01, 0.01, 0.01, 0.01  # insertion / deletion error, impulse noise, permanent frequency disturbance
prob = [0.046, 0.041, 0.036, 0.031, 0.026, 0.021, 0.016, 0.011, 0.006, 0.001]  # background noise
max_ins, max_len = 1, n+2  # maximum insertions in one slot, maximum column length

#-----------------------------------model setting-------------------------------------------------------#

words_gf, labels = np.empty((0,max_len), dtype=np.uint8), np.empty((0,), dtype=np.uint8)

for i in range(len(prob)):
    name = f'./perm_plc_syn_data/trainw_gf{n,code_size}{p_i, p_d, p_im, p_pfd, prob[i]}.npz'
    words_gf = np.concatenate((words_gf, sparse.load_npz(name).toarray().astype(np.uint8)), axis=0)
 
    name = f'./perm_plc_syn_data/trainl_syn{n,code_size}{p_i, p_d, p_im, p_pfd, prob[i]}.npz'
    labels = np.concatenate((labels, np.squeeze(sparse.load_npz(name).toarray().astype(np.uint8))), axis=0)
    
    name = f'./perm_plc_syn_data/trainw_gf_add{n,code_size}{p_i, p_d, p_im, p_pfd, prob[i]}.npz'
    words_gf = np.concatenate((words_gf, sparse.load_npz(name).toarray().astype(np.uint8)), axis=0)
 
    name = f'./perm_plc_syn_data/trainl_syn_add{n,code_size}{p_i, p_d, p_im, p_pfd, prob[i]}.npz'
    labels = np.concatenate((labels, np.squeeze(sparse.load_npz(name).toarray().astype(np.uint8))), axis=0)
    
labels = np.where(labels>7, 7, labels)

inputs = Input(shape=(max_len,), name='input')
x = layers.Embedding(64, 8, input_length=max_len)(inputs)
x = layers.Flatten()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(8, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)
model.summary()

# model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=False), optimizer='adam')

scce = losses.SparseCategoricalCrossentropy(from_logits=False) 
def loss_new(labels, outputs):
    loss1 = tf.keras.losses.MSE(labels, tf.reduce_sum(outputs*tf.range(8, dtype=tf.float32), axis=-1))
    loss2 = scce(labels, outputs)
    return 0.1*loss1 + 0.9*loss2 
    
model.compile(loss=loss_new, optimizer='adam')
          
#-----------------------------------training phase-------------------------------------------------------#
epoch = 5
p_i, p_d, p_im, p_pfd = 0.001, 0.001, 0, 0  # insertion / deletion error, impulse noise, permanent frequency disturbance
test_err = np.zeros((len(prob)), dtype=float)

for r in range(1, 5):
    model.fit(words_gf, labels, batch_size=200, epochs=epoch, verbose=2)    
    model.save(f'./perm_plc_syn{n, code_size}_e8_mlp256-128_{r*epoch}')

    for i in range(len(prob)):
        name = f'./perm_plc_syn_data/testw_gf{n,code_size}{p_i, p_d, p_im, p_pfd, prob[i]}.npz'
        test_words_gf = sparse.load_npz(name).toarray().astype(np.uint8)

        name = f'./perm_plc_syn_data/testl_syn{n,code_size}{p_i, p_d, p_im, p_pfd, prob[i]}.npz'
        test_labels = np.squeeze(sparse.load_npz(name).toarray().astype(np.uint8))
        
        test_pred = np.argmax(model.predict(test_words_gf, batch_size=200), axis=-1)
    
        test_err[i] = 1.0 - np.sum(test_pred == test_labels) / test_words_gf.shape[0]

    fl = open(f'./perm_plc_syn_data/teste{n,code_size}{p_i, p_d, p_im, p_pfd}_{method}.txt','a')
    np.savetxt(fl, test_err, fmt="%.6f", delimiter=' ')
    fl.close()
