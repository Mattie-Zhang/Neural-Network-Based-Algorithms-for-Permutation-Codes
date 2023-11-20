import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # No display of info, 0,1,2,3

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import Input, Model
from scipy import sparse

#-----------------------------------parameters-------------------------------------------------------#

n, code_size, method = 6, 56, 'eda'
p_i, p_d, p_im, p_pfd = 0.01, 0.01, 0.01, 0.01  # insertion / deletion error, impulse noise, permanent frequency disturbance
prob = [0.046, 0.041, 0.036, 0.031, 0.026, 0.021, 0.016, 0.011, 0.006, 0.001]  # background noise
max_ins, max_len = 1, n+2 # maximum insertions in one slot, maximum column length

#-----------------------------------model setting-------------------------------------------------------#

heading, units = 64, 128

words_gf, labels = np.zeros((0, max_len), dtype=np.uint8), np.zeros((0, n), dtype=np.uint8)

for i in range(len(prob)):
    name = f'./perm_plc_dec_data/trainw_gf{n,code_size}{p_i, p_d, p_im, p_pfd, prob[i]}.npz'
    words_gf = np.concatenate((words_gf, sparse.load_npz(name).toarray().astype(np.uint8)), axis=0)
    
    name = f'./perm_plc_dec_data/trainl{n,code_size}{p_i, p_d, p_im, p_pfd, prob[i]}.npz'
    labels = np.concatenate((labels, sparse.load_npz(name).toarray().astype(np.uint8)), axis=0)

labels = np.concatenate((heading * np.ones((labels.shape[0],1), dtype=np.uint8), labels), axis=1)  # add 64 at the beginning of each output

class Encoder(tf.keras.layers.Layer):

    def __init__(self, emb):
        super(Encoder, self).__init__()
        self.emb = emb
        self.brnn = tf.keras.layers.Bidirectional(merge_mode='sum', layer=tf.keras.layers.LSTM(units, return_sequences=True))
        
    def call(self, enc_in):
        enc_out = self.emb(enc_in)
        enc_out = self.brnn(enc_out)        
        return enc_out

class CrossAttention(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=1, key_dim=units)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, dec_out, enc_out):
        attn_out = self.mha(query=dec_out, value=enc_out)
        dec_out = self.add([dec_out, attn_out])
        dec_out = self.layernorm(dec_out)
        return dec_out
        
class Decoder(tf.keras.layers.Layer):

    def __init__(self, emb):
        super(Decoder, self).__init__()
        self.emb = emb
        self.rnn = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)
        self.attention = CrossAttention()
        self.output_layer = tf.keras.layers.Dense(n)
    
    def call(self, enc_out, dec_in, hidden, cell):  # None causes creation of zero-filled initial state tensors
    
        dec_out = self.emb(dec_in)
        dec_out, hidden, cell = self.rnn(dec_out, initial_state=[hidden, cell])
        dec_out = self.attention(dec_out, enc_out)
        dec_out = self.output_layer(dec_out)
        return dec_out, hidden, cell
  
class Translator(tf.keras.Model):

    def __init__(self, ):
        super().__init__()
        self.emb = tf.keras.layers.Embedding(65, 8)  # use 64 to denote the beginning of each output
        self.encoder = Encoder(self.emb)
        self.decoder = Decoder(self.emb)

    def call(self, data):
        
        hidden, cell = tf.zeros([200, units]), tf.zeros([200, units])

        enc_in, dec_in = data
        enc_out = self.encoder(enc_in)
        dec_out, _, _ = self.decoder(enc_out, dec_in, hidden, cell)
        return dec_out
        
def process(enc_in, dec):
    dec_in, dec_out = dec[:,:-1], dec[:,1:]
    return (enc_in, dec_in), dec_out

train_data = (tf.data.Dataset.from_tensor_slices((words_gf, labels))).shuffle(words_gf.shape[0]).batch(200).map(process)

model = Translator()
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
# model.summary()

#-----------------------------------training phase-------------------------------------------------------#

model.fit(train_data, batch_size=200, epochs=5, verbose=1)

model.save(f'./perm_plc_dec{n,code_size}_e8_eda{units}_5')

# model = tf.keras.models.load_model(f'./perm_plc_model/perm_plc_dec{n,code_size}_e8_eda{units}_5', compile=False)
# weights = model.get_weights()  # save the trainable parameters directly
# np.save(f'./perm_plc_dec_eda{n,code_size}_parameters.npy', np.array(weights, dtype=object), allow_pickle=True)

words_gf, labels = [], []

prob_set = [[0.0005,0.0005,0.0005,0.0005],[0.001,0.001,0.001,0.001],[0.005,0.005,0.005,0.005],[0.01,0.01,0.01,0.01],
[0.0005,0.0005,0,0],[0.001,0.001,0,0],[0.005,0.005,0,0],[0.01,0.01,0,0]]

test_err = np.zeros((len(prob)), dtype=float)
batch_size = 10000

for r in range(8):

    p_i, p_d, p_im, p_pfd = prob_set[r][0], prob_set[r][1], prob_set[r][2], prob_set[r][3]  # insertion/deletion, impulse noise, permanent frequency disturbance
    
    for i in range(len(prob)):
        name = f'./perm_plc_dec_data/testw_gf{n,code_size}{p_i, p_d, p_im, p_pfd, prob[i]}.npz'
        test_words_gf = sparse.load_npz(name).toarray().astype(np.uint8)

        name = f'./perm_plc_dec_data/testl{n,code_size}{p_i, p_d, p_im, p_pfd, prob[i]}.npz'
        test_labels = sparse.load_npz(name).toarray().astype(np.uint8) 
        
        test_pred_labels = np.zeros((test_labels.shape[0], n), dtype=np.uint8)
    
        for j in range(int(test_words_gf.shape[0] / batch_size)):

            dec_out = heading * np.ones((batch_size, 1), dtype=np.uint8)
            hidden, cell = tf.zeros([batch_size, units]), tf.zeros([batch_size, units])
            enc_out = model.encoder(test_words_gf[(j*batch_size):((j+1)*batch_size)]) 
         
            for k in range(n):
        
                dec_out, hidden, cell = model.decoder(enc_out, dec_out, hidden, cell)
                dec_out = tf.argmax(dec_out, axis=-1)
                test_pred_labels[(j*batch_size):((j+1)*batch_size), k] = np.squeeze(dec_out)

        test_err[i] = 1.0 - (test_pred_labels == test_labels).all(axis=-1).sum() / test_words_gf.shape[0]  
    
    fl = open(f'./perm_plc_dec_data/teste{n,code_size}{p_i, p_d, p_im, p_pfd}_{method}.txt','w')
    np.savetxt(fl, test_err, fmt="%.6f", delimiter=',')
    fl.close()
        
    
    
    
    
    
    





