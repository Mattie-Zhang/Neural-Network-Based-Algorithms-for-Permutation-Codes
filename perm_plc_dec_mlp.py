import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # No display of info, 0,1,2,3

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import Input, Model
from scipy import sparse

#-----------------------------------parameters-------------------------------------------------------#

n, code_size, method = 7, 360, 'mlp'
p_i, p_d, p_im, p_pfd = 0.01, 0.01, 0.01, 0.01  # insertion / deletion error, impulse noise, permanent frequency disturbance
prob = [0.046, 0.041, 0.036, 0.031, 0.026, 0.021, 0.016, 0.011, 0.006, 0.001]  # background noise
max_ins, max_len = 1, n+2  # maximum insertions in one slot, maximum column length

#-----------------------------------model setting-------------------------------------------------------#

words_gf, labels = np.empty((0, max_len), dtype=np.uint8), np.empty((0, n), dtype=np.uint8)

for i in range(len(prob)):
    name = f'./perm_plc_dec_data/trainw_gf{n,code_size}{p_i, p_d, p_im, p_pfd, prob[i]}.npz'
    words_gf = np.concatenate((words_gf, sparse.load_npz(name).toarray().astype(np.uint8)), axis=0)

    name = f'./perm_plc_dec_data/trainl{n,code_size}{p_i, p_d, p_im, p_pfd, prob[i]}.npz'
    labels = np.concatenate((labels, sparse.load_npz(name).toarray().astype(np.uint8)), axis=0)
    
labels_mul = [labels[:,i] for i in range(n)]

model = tf.keras.Sequential([
    layers.Embedding(128,8,input_length=max_len),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dropout(rate=0.1),
])

inputs = Input(shape=(max_len,), name='input')
x = model(inputs)
output0 = layers.Dense(n, activation='softmax')(x)
output1 = layers.Dense(n, activation='softmax')(x)
output2 = layers.Dense(n, activation='softmax')(x)
output3 = layers.Dense(n, activation='softmax')(x)
output4 = layers.Dense(n, activation='softmax')(x)
output5 = layers.Dense(n, activation='softmax')(x)
output6 = layers.Dense(n, activation='softmax')(x)

model = Model(inputs=inputs, outputs=[output0,output1,output2,output3,output4,output5,output6])  # 
model.summary()

model.compile(loss=[losses.SparseCategoricalCrossentropy(from_logits=False),
                losses.SparseCategoricalCrossentropy(from_logits=False),
                losses.SparseCategoricalCrossentropy(from_logits=False),
                losses.SparseCategoricalCrossentropy(from_logits=False),
                losses.SparseCategoricalCrossentropy(from_logits=False),
                losses.SparseCategoricalCrossentropy(from_logits=False),
                losses.SparseCategoricalCrossentropy(from_logits=False)
                    ],
            loss_weights = [1,1,1,1,1,1,1],  #
            optimizer = 'adam',
            metrics = ['accuracy'])
            
#-----------------------------------training phase-------------------------------------------------------#
epoch = 2
p_i, p_d, p_im, p_pfd = 0.001, 0.001, 0, 0  # insertion / deletion error, impulse noise, permanent frequency disturbance
test_err = np.zeros((len(prob)), dtype=float)

for r in range(5):
    model.fit(words_gf, [labels_mul[0],labels_mul[1],labels_mul[2],labels_mul[3],labels_mul[4],labels_mul[5],labels_mul[6]],
            batch_size=200, epochs=epoch, verbose=2)
          
    model.save(f'./perm_plc_dec{n,code_size}_e8_mlp512-512-512_{r*epoch}')

    for i in range(len(prob)):
        name = f'./perm_plc_dec_data/testw_gf{n,code_size}{p_i, p_d, p_im, p_pfd, prob[i]}.npz'
        test_words_gf = sparse.load_npz(name).toarray().astype(np.uint8)

        name = f'./perm_plc_dec_data/testl{n,code_size}{p_i, p_d, p_im, p_pfd, prob[i]}.npz'
        test_labels = sparse.load_npz(name).toarray().astype(np.uint8)

        test_pred = model.predict(test_words_gf, batch_size=200)  # shape[n,test_num,n]
        test_pred = np.transpose(np.argmax(test_pred, axis=-1))
        test_err[i] = 1.0 - (test_pred == test_labels).all(axis=-1).sum() / test_words_gf.shape[0]

    fl = open(f'./perm_plc_dec_data/teste{n,code_size}{p_i, p_d, p_im, p_pfd}_{method}.txt','a')
    np.savetxt(fl, test_err, fmt="%.6f", delimiter=' ')
    fl.close()

