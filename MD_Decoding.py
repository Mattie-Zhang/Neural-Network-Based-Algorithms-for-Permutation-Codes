import numpy as np
import itertools

class MDD_Class:
    def __init__(self, code):
    
        self.code_size, self.n = code.shape 
    
        self.code_mats = np.zeros((self.code_size*self.n, self.n), dtype=np.uint8)
        self.code_mats[np.arange(self.code_size*self.n),code.flatten()] = 1
        self.code_mats = np.transpose(np.reshape(self.code_mats, (self.code_size,self.n,self.n)), (0,2,1))

        self.code_submats_1 = np.zeros((0, self.n, self.n-1), dtype=np.uint8)  # all n*(n-1) sub-matrices of code_mats
        for sub in itertools.combinations(np.arange(self.n), self.n-1):
            self.code_submats_1 = np.concatenate((self.code_submats_1, self.code_mats[:, :, sub]), axis=0)
            self.code_submats_idx_1 = np.tile(np.arange(self.code_size), int(self.code_submats_1.shape[0]/self.code_size))
    
        self.code_submats_2 = np.zeros((0, self.n, self.n-2), dtype=np.uint8)  # all n*(n-2) sub-matrices of code_mats
        for sub in itertools.combinations(np.arange(self.n), self.n-2):
            self.code_submats_2 = np.concatenate((self.code_submats_2, self.code_mats[:, :, sub]), axis=0)
            self.code_submats_idx_2 = np.tile(np.arange(self.code_size), int(self.code_submats_2.shape[0]/self.code_size))
    
    def Code_Submats(self, flag): 
        if flag == self.n-2:
            return self.code_submats_2, self.code_submats_idx_2
        elif flag == self.n-1:
            return self.code_submats_1, self.code_submats_idx_1
        else:
            return self.code_mats, np.arange(self.code_size)
    
    def Word_Submats(self, word_sub):
        word_submats = np.zeros((0, self.n, self.n), dtype=np.uint8)
        for sub in itertools.combinations(np.arange(word_sub.shape[1]), self.n):
            word_submats = np.concatenate((word_submats, word_sub[None, :, sub]), axis=0)
        return word_submats
    
    # maximum likelyhood decoding without i/d
    def md_decoding0(self, words):
        lab_idx_pred = []
        for i in range(words.shape[0]):
            tmp = (words[i, :, :self.n] != self.code_mats).sum(axis=2).sum(axis=1)
            lab_idx_pred.append(np.argmin(tmp))
        return np.array(lab_idx_pred)

    # maximum likelyhood decoding with i/d
    def md_decoding(self, words):
        lab_idx_pred = []
        for i in range(words.shape[0]):
            max_idx = np.max(np.nonzero(words[i])[1]) + 1  # index of the largest nonzero column
            if max_idx == self.n-2 or max_idx == self.n-1 or max_idx == self.n:
                code_submats, code_submats_idx = self.Code_Submats(max_idx)
                tmp = (words[i, :, :max_idx] != code_submats).sum(axis=2).sum(axis=1)
                lab_idx_pred.append(code_submats_idx[np.argmin(tmp)])
            
            elif max_idx == self.n+1 or max_idx == self.n+2:
                word_submats = self.Word_Submats(words[i, :, :max_idx])
                code_tmp = np.repeat(self.code_mats[:, None, :, :], word_submats.shape[0], axis=1)
                tmp = (word_submats != code_tmp).sum(axis=3).sum(axis=2)
                lab_idx_pred.append(np.argmin(np.min(tmp, axis=-1)))
            
            else:
                lab_idx_pred.append(self.code_size)
        return np.array(lab_idx_pred)
