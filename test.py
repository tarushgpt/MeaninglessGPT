import numpy as np
from util import Util
import config

class Test:

    def __init__(self):
        self.d_model = config.d_model
        self.d_k = config.d_k
        self.d_v = config.d_v
        self.d_ff = config.d_ff
        self.lr = config.lr
        self.util = Util()


        self.E = np.load("weights/E.npy")
        self.Wq = np.load("weights/Wq.npy")
        self.Wk = np.load("weights/Wk.npy")
        self.Wv = np.load("weights/Wv.npy")
        self.W1 = np.load("weights/W1.npy")
        self.b1 = np.load("weights/b1.npy")
        self.W2 = np.load("weights/W2.npy")
        self.b2 = np.load("weights/b2.npy")
        self.Wo = np.load("weights/Wo.npy")
        self.bo = np.load("weights/bo.npy")


    def test(self, vocabulary):
        
        while True:
            input_str = input("Enter your input here, or press 'Enter' to quit: ")
            print("\n")
            if input_str == "":
                break
            chunked = []
            for i in input_str: chunked.append(i)
            for i in range(len(chunked)):
                chunked[i] = vocabulary.index(chunked[i])
            t = len(chunked)

            X = []
            for i in range(t):
                index = chunked[i]
                X.append(self.E[index])
            X = np.array(X) 

            Q = X @ self.Wq 
            K = X @ self.Wk 
            V = X @ self.Wv

            scores = Q @ K.T 
            mask = []
            for i in range(t):
                row = []
                for j in range(t):
                    if j <= i: row.append(0.0)
                    else: row.append(-np.inf)
                mask.append(row)
            mask = np.array(mask) 

            scores_norm = scores / np.sqrt(self.d_k) + mask
            scores_soft = self.util.softmax(scores_norm) 

            output = scores_soft @ V 

            h1 = output @ self.W1 + self.b1 
            h2 = self.util.relu(h1) 
            h3 = h2 @ self.W2 + self.b2 

            logits = h3[-1] @ self.Wo + self.bo 
            probability = self.util.softmax_row(logits) 
            print("MeaninglessGPT: " + vocabulary[np.argmax(probability)] + "\n")

