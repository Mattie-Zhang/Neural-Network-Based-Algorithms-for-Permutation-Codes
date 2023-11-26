import numpy as np


def gen_words(code, labels, p_set, max_ins, max_len):

    code_size, n = code.shape
    p_i, p_d, p_im, p_pfd, p_bg = p_set
    num, max_col_len = labels.shape[0], n+max_ins*(n+1)

    i_err = np.random.choice(a=2, size=(num*(n+1)), replace=True, p=[1-p_i, p_i])
    d_err = np.random.choice(a=2, size=(num,n), replace=True, p=[1-p_d, p_d])
    
    #---------------------------------------------------------
    lab_syn = np.sum(i_err.reshape(num, n+1), axis=-1) + np.sum(d_err, axis=-1)  # the sum of number of insertions and deletions
    #---------------------------------------------------------
    
    i_err[np.nonzero(i_err)[0]] = np.random.randint(1, n+1, size=(len(np.nonzero(i_err)[0]),))
    
    tmp = np.zeros((num, max_col_len), dtype=int)
    tmp[:,::2], tmp[:,1::2] = i_err.reshape(num,n+1)-1, np.where(d_err==1, -1, labels)

    col_len = np.count_nonzero(tmp+1, axis=1)
    for i in range(num):
        tmp[i,:col_len[i]] = tmp[i,np.where(tmp[i]!=-1)]
    tmp = tmp[:,:max_len]

    tmp = tmp.flatten()
    words = np.zeros((tmp.size, n), dtype=int)
    words[np.arange(tmp.size),tmp] = 1
    words = np.transpose(np.reshape(words, (num,max_len,n)), (0,2,1))
    
    bg_noise = np.random.choice(a=2, size=(num,n,max_len), replace=True, p=[1-p_bg,p_bg])
    words = (words + bg_noise) % 2
    
    im_noise = np.random.choice(a=2, size=(num,max_len,), replace=True, p=[1-p_im,p_im])
    im_noise = np.repeat(im_noise[:,None,:], n, axis=1)
    words = np.logical_or(words, im_noise)
    
    pdf_noise = np.random.choice(a=2, size=(num,n,), replace=True, p=[1-p_pfd,p_pfd])
    pdf_noise = np.repeat(pdf_noise[:,:,None], max_len, axis=-1)
    words = np.logical_or(words, pdf_noise)

    for i in range(num):
        words[i,:,col_len[i]:] = 0
    return words, lab_syn
    
    
def gen_syn_arr(code, num, slots, seq_len, p_set):
    
    code_size, n = code.shape
    p_i, p_d, p_im, p_pfd, p_bg = p_set
    
    seq_len_max = 2*n*slots+1
    seq_idx = np.random.randint(code_size, size=(num,slots))
    seq = code[seq_idx].reshape(num, n*slots)  # original sequence
    
    i_err = np.random.choice(a=2, size=(num*(n*slots+1)), replace=True, p=[1-p_i, p_i])
    d_err = np.random.choice(a=2, size=(num, n*slots), replace=True, p=[1-p_d, p_d])

    #---------------------------------------------------------
    slot_idx = np.empty((num, seq_len_max), dtype=int)  # label the columns belong to which slot
    slot_idx_tmp = np.repeat(np.repeat(np.arange(slots), n)[None, :], num, axis=0)
    slot_idx[:, ::2] = np.where(i_err==1, slots, -1).reshape(num,n*slots+1)  # index the insertion as `slots'
    slot_idx[:, 1::2] = np.where(d_err==1, -1, slot_idx_tmp)
    #---------------------------------------------------------

    i_err[np.nonzero(i_err)[0]] = np.random.randint(1, n+1, size=(len(np.nonzero(i_err)[0]),))
    seq_tmp = np.empty((num, seq_len_max), dtype=int)
    seq_tmp[:, ::2] = i_err.reshape((num, n*slots+1)) - 1
    seq_tmp[:, 1::2] = np.where(d_err==1, -1, seq)

    eff_len = np.count_nonzero(seq_tmp+1, axis=1)  # assume the eff_len is known to decoder
     
    for i in range(num):  # remove -1
        seq_tmp[i,:eff_len[i]] = seq_tmp[i, np.where(seq_tmp[i]!=-1)]
        slot_idx[i,:eff_len[i]] = slot_idx[i, np.where(slot_idx[i]!=-1)]
    seq_tmp, slot_idx = seq_tmp[:,:seq_len], slot_idx[:,:seq_len]

    arr = np.zeros((num*seq_len, n), dtype=int)  # change to array and add noises
    arr[np.arange(num*seq_len), seq_tmp.flatten()] = 1
    arr = np.transpose(np.reshape(arr, (num,seq_len,n)), (0,2,1))
    
    bg_noise = np.random.choice(a=2, size=(num,n,seq_len), replace=True, p=[1-p_bg,p_bg])
    arr = (arr + bg_noise) % 2   

    im_noise = np.random.choice(a=2, size=(num,seq_len), replace=True, p=[1-p_im,p_im])
    im_noise = np.repeat(im_noise[:,None,:], n, axis=1)
    arr = np.logical_or(arr, im_noise)

    pdf_noise = np.random.choice(a=2, size=(num,n,), replace=True, p=[1-p_pfd,p_pfd])
    pdf_noise = np.repeat(pdf_noise[:,:,None], seq_len, axis=-1)
    arr = np.logical_or(arr, pdf_noise)
    
    for i in range(num):
        arr[i,:,eff_len[i]:], slot_idx[i,eff_len[i]:] = 0, -1
    
    eff_len = np.minimum(eff_len, seq_len)
    
    return seq_idx, seq, arr, slot_idx, eff_len


def find_gf_one(tmp, gf_b, n, q):

    tmp = np.transpose(tmp, (0,2,1)).reshape(-1, n)
    tmp = np.repeat(tmp[:, None, :], q, axis=-2)
    tmp = np.nonzero((tmp == gf_b).all(axis=-1))[1]
    return tmp
    

def find_gf(words):  # 6:x^6+x+1, 7:x^7+x+1
 
    n, q = words.shape[1], np.power(2, words.shape[1])
    gf_b = np.loadtxt(f'./gf_b{q}.txt')
     
    rounds = int(np.ceil(words.shape[0] / 1000000))
    words_gf = np.zeros((0, words.shape[-1]), dtype=np.uint8)
    for r in range(rounds):    
        tmp = find_gf_one(words[r*1000000:(r+1)*1000000], gf_b, n, q).reshape(-1, words.shape[-1])
        words_gf = np.concatenate((words_gf, tmp), axis=0)
    return words_gf
    

def syn_simple(words, col_len):
    
    return col_len + words.shape[1] - 2*np.count_nonzero(np.sum(words, axis=-1), axis=-1) 


def syn_mlp(model_syn, words):
    
    words_gf = find_gf(words)
    return np.argmax(model_syn.predict(words_gf, batch_size=10000, verbose=2), axis=-1)

 

