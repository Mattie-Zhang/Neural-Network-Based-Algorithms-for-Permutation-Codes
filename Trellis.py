import numpy as np
from collections import Counter
from perm_functions import find_gf
import tensorflow as tf

class Trellis_Class:
    def __init__(self, mode, n, max_len, slots, scope, arr, eff_len, slot_idx):
        self.mode, self.n, self.max_len, self.slots, self.scope = mode, n, max_len, slots, scope
        self.arr, self.slot_idx, self.eff_len = arr, slot_idx, eff_len
        
        self.nb = 5
        
        self.words, self.lab_syn, self.col_len, self.count = [], [], [], 0
        self.flag = -np.ones((self.slots+1, 4*self.scope+1, 5), dtype=int)  # store the index of possible words
        self.gen_words_flag()
        
        self.words = np.array(self.words, dtype=np.uint8)
        self.col_len = np.array(self.col_len, dtype=np.uint8)
        
        if self.mode == 'syn':
            self.lab_syn = np.array(self.lab_syn).astype(np.uint8)
 
    def gen_words_flag(self):  # i denotes the error type of the word in previous time slot
        for j in range(-2,3):
            self.assign_words_flag(1,j,j)
        for t in range(2, self.scope+1):
            for j in range(-2*t, 2*t+1):
                if j>=2*t-3:
                    for i in range(j-2*t+2,3):
                        self.assign_words_flag(t,j,i)
                elif j<=-2*t+3:
                    for i in range(-2, j+2*t-1):
                        self.assign_words_flag(t,j,i)
                else:
                    for i in range(-2,3):
                        self.assign_words_flag(t,j,i)
        for t in range(self.scope+1, self.slots+1):
            for j in range(-2*self.scope, 2*self.scope+1):
                    if j>=2*self.scope-1:
                        for i in range(j-2*self.scope,3):
                            self.assign_words_flag(t,j,i)
                    elif j<=-2*self.scope+1:
                        for i in range(-2, j+2*self.scope+1):
                            self.assign_words_flag(t,j,i)
                    else:
                        for i in range(-2,3):
                            self.assign_words_flag(t,j,i)
        
    def assign_words_flag(self, t, j, i):
        start, end = (t-1)*self.n+j-i, np.minimum(t*self.n+j, self.eff_len)
        if start<end:
            words_tmp = np.zeros((self.n, self.max_len), dtype=np.uint8)
            words_tmp[:, :(end-start)] = self.arr[:, start:end]
            self.words.append(words_tmp)
            self.col_len.append(end-start)
            self.flag[t,j,i] = self.count 
            self.count += 1
            
            if self.mode == 'syn':
                _, indices = np.unique(self.slot_idx[start:end], return_inverse=True)            
                self.lab_syn.append((end-start) + self.n - 2*np.max(np.bincount(indices))) 
        
    def assign_epsilon(self, syn_num):
        epsilon = -np.ones((self.slots+1, 4*self.scope+1), dtype=int)
        words_min = np.zeros((self.slots+1, self.nb), dtype=int)  # store the index of most possible codewords
            
        epsilon[0,0] = 0
        for t in range(1,self.slots+1):
            beta = []
            for j in range(-2*min(t,self.scope), 2*min(t,self.scope)+1):
                alpha = []
                for i in range(-2,3):
                    if self.flag[t,j,i] != -1:
                        tmp = epsilon[t-1,j-i] + syn_num[self.flag[t,j,i]]  # = alpha[t,j,i]
                        alpha.append(tmp)
                        beta.append([tmp,self.flag[t,j,i]])
                if len(alpha) != 0:
                    epsilon[t,j] = min(alpha)
            beta = np.array(beta)
            words_min[t] = beta[np.argsort(beta[:,0])[:self.nb], 1]
        return words_min
            
            
    def count_err_mdd(self, mdd, seq_idx, words, words_min):
    
        seq_pred = mdd.md_decoding(words[words_min[1:].flatten()]).reshape(self.slots, self.nb)
        seq_pred = [np.bincount(seq_pred[i]) for i in range(self.slots)]
        seq_idx_pred = np.array([np.argmax(seq_pred[i]) for i in range(self.slots)])
        return (seq_idx_pred == seq_idx).sum()


    def count_err_mlp(self, model_dec, seq, words, words_min):

        words_gf = find_gf(words[words_min[1:].flatten()])
        words_pred = np.argmax(model_dec.predict(words_gf,batch_size=10000,verbose=2), axis=-1).T.reshape(self.slots,self.nb,self.n)
    
        seq_pred = np.zeros((self.slots, self.n), dtype=int)
        for t in range(self.slots):
            seq_pred[t] = self.find_most_freq(words_pred[t])
        return (seq_pred == seq.reshape(self.slots, self.n)).all(axis=-1).sum()
    
    def count_err_eda(self, model_dec, seq, words, words_min, units, heading):

        words_gf = find_gf(words[words_min[1:].flatten()])

        dec_out = heading*np.ones((words_gf.shape[0], 1), dtype=np.uint8)        
        hidden, cell = tf.zeros([words_gf.shape[0], units]), tf.zeros([words_gf.shape[0], units])
        enc_out = model_dec.encoder(words_gf)
        
        words_pred = np.zeros((words_gf.shape[0], self.n), dtype=np.uint8)
         
        for k in range(self.n):
        
            dec_out, hidden, cell = model_dec.decoder(enc_out, dec_out, hidden, cell)
            dec_out = tf.argmax(dec_out, axis=-1)
            words_pred[:, k] = np.squeeze(dec_out)
            
        words_pred = words_pred.reshape(self.slots,self.nb,self.n)

        seq_pred = np.zeros((self.slots, self.n), dtype=int)
        for t in range(self.slots):
            seq_pred[t] = self.find_most_freq(words_pred[t])
        return (seq_pred == seq.reshape(self.slots, self.n)).all(axis=-1).sum()


    def find_most_freq(self, tmp):
    
        seq_dict = Counter([tuple(t.tolist()) for t in tmp])
        return np.asarray(max(seq_dict, key=seq_dict.get))
            
            

