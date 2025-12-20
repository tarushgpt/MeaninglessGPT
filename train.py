import concurrent.futures
import numpy as np
import os, config
from util import Util

class Train:

    def __init__(self):
        self.d_model = config.d_model
        self.d_k = config.d_k
        self.d_v = config.d_v
        self.d_ff = config.d_ff
        self.lr = config.lr
        self.training_steps = config.training_steps
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

    def train(self, example, vocabulary):

        self.v_len = len(vocabulary) 

        examples = []

        for step in range(self.training_steps):
            trained = self.train_step(example, vocabulary)
            
            self.E -= self.lr * trained["dE"]
            self.Wq -= self.lr * trained["dWq"]
            self.Wk -= self.lr * trained["dWk"]
            self.Wv -= self.lr * trained["dWv"]
            self.W1 -= self.lr * trained["dW1"]
            self.b1 -= self.lr * trained["db1"]
            self.W2 -= self.lr * trained["dW2"]
            self.b2 -= self.lr * trained["db2"]
            self.Wo -= self.lr * trained["dWo"]
            self.bo -= self.lr * trained["dbo"]



        np.save("weights/E.npy", self.E)
        np.save("weights/Wq.npy", self.Wq)
        np.save("weights/Wk.npy", self.Wk)
        np.save("weights/Wv.npy", self.Wv)
        np.save("weights/W1.npy", self.W1)
        np.save("weights/b1.npy", self.b1)
        np.save("weights/W2.npy", self.W2)
        np.save("weights/b2.npy", self.b2)
        np.save("weights/Wo.npy", self.Wo)
        np.save("weights/bo.npy", self.bo)


        print("Training completed. continuing on to testing.")


    def train_step(self, example, vocabulary):
        input_str = example.split("->")[0]
        next_token = example.split("->")[1]


        E = self.E.copy()
        Wq = self.Wq.copy()
        Wk = self.Wk.copy()
        Wv = self.Wv.copy()
        W1 = self.W1.copy()
        b1 = self.b1.copy()
        W2 = self.W2.copy()
        b2 = self.b2.copy()
        Wo = self.Wo.copy()
        bo = self.bo.copy()


        
        #character based tokenization 
        chunked = []
        for i in input_str: chunked.append(i)

        #mapping input to token ids
        for i in range(len(chunked)):
            chunked[i] = vocabulary.index(chunked[i])
        t = len(chunked)

        #creating input matrix x
        X = []
        for i in range(t):
            index = chunked[i]
            X.append(E[index]) #using the token id to find the corresponding embeddings for the token
        X = np.array(X) #dimensions of X are (t * self.d_model)

        #Attention Block

        Q = X @ Wq #dimensions of Wq are (self.d_model * self.d_k), so dimensions of Q are (t * self.d_k)
        K = X @ Wk #dimensions of Wk are (self.d_model * self.d_k), so dimensions of Q are (t * self.d_k)
        V = X @ Wv #dimensions of Wv are (self.d_model * self.d_v), so dimensions of Q are (t * self.d_v)

        scores = Q @ K.T #dimenions of scores are (t * t)

        #initializing casual mask to avoid looking at future tokens and "cheating" during training
        mask = []
        for i in range(t):
            row = []
            for j in range(t):
                if j <= i: row.append(0.0)
                else: row.append(-np.inf)
            mask.append(row)
        mask = np.array(mask) #dimensions of mask is also (t * t)

        scores_norm = scores / np.sqrt(self.d_k) + mask
        scores_soft = self.util.softmax(scores_norm) #still (t * t)

        output = scores_soft @ V #reminder that dimensions of V are (t * self.d_v), so output is (t * self.d_v)

        #FFN

        h1 = output @ W1 + b1 #dimensions of W1 is (self.d_v * dff) and dimensions of b1 is (self.d_ff,), so h1 is (t * self.d_ff)
        h2 = self.util.relu(h1) #dimensions do not change
        h3 = h2 @ W2 + b2 #dimensions of W2 is (self.d_ff * self.d_model) and dimensions of b2 is (self.d_model,), so h3 is (t, self.d_model)

        logits = h3[-1] @ Wo + bo #dimensions of W0 is (self.d_model, self.v_len) and b0 are (self.v_len,), so dimensions of logits are (self.v_len,)
        probability = self.util.softmax_row(logits) #dimensions do not change

        #end of forward pass- we would argmax to retrieve the value here
        #print(vocabulary[np.argmax(probability)])


        #correct token
        correct_token_id = vocabulary.index(next_token)
        
        #Backward pass (weight/bias update)

        #CE loss
        loss = -np.log(probability[correct_token_id])

        #derivative of CE
        #initializing a one-hot vector based on Kronecker delta
        e_y = np.zeros_like(probability)
        e_y[correct_token_id] = 1

        #solving for the derivative wrt logits
        dL_dlogits = probability - e_y

        #using an outer product to find the derivative with respect to the output weights in FFN
        dL_dw0 = np.outer(h3[-1], dL_dlogits)
        dL_db0 = dL_dlogits

        #using a backpropogation rule to find the derivative wrt h3_last (used in token prediction)
        dL_dh3_last = dL_dlogits @ Wo.T

        #creating the derivative wrt h3 by setting the other ones to zero (does not affect the final output)
        dL_dh3 = np.zeros_like(h3)
        dL_dh3[-1] = dL_dh3_last

        #finds d_dW2 and d_db2 for the second layer of the FFN
        dL_dW2 = h2.T @ dL_dh3
        dL_db2 = dL_dh3.sum(axis=0)
        dL_dh2 = dL_dh3 @ W2.T

        #finds d_dW1 and d_db1 for the first layer of the FFN
        dL_dh1 = dL_dh2 * self.util.reluprime(h1)
        dL_dW1 = output.T @ dL_dh1 
        dL_db1 = dL_dh1.sum(axis=0)

        #compute loss gradient wrt output weights
        dL_doutput = dL_dh1 @ W1.T

        #compute loss gradients wrt value tensor
        dL_dV = scores_soft.T @ dL_doutput
        #compute loss gradients wrt softmaxed scores
        dL_dscoressoft = dL_doutput @ V.T

        #compute loss gradients wrt non softmaxed (CM + 1/sqrt(dk)) scores
        dL_dscores_norm = np.zeros_like(dL_dscoressoft)

        for i in range(t):
            p = scores_soft[i]         
            g = dL_dscoressoft[i]       
            dot = np.sum(g * p)
            dL_dscores_norm[i] = p * (g - dot)

        #compute gradient wrt scores
        dL_dscores = 1/np.sqrt(self.d_k) * dL_dscores_norm
        #compute loss gradient wrt query tensor
        dL_dQ = dL_dscores @ K
        #compute loss gradient wrt transpose of key
        dL_dKT = Q.T @ dL_dscores
        #compute loss gradient wrt key
        dL_dK = dL_dKT.T

        #compute the gradients wrt the weights of query, key, and value
        dL_dWv = X.T @ dL_dV
        dL_dWk = X.T @ dL_dK
        dL_dWq = X.T @ dL_dQ

        #find the loss gradient wrt input X
        dL_dX = dL_dQ @ Wq.T + dL_dK @ Wk.T + dL_dV @ Wv.T
        
        #finding final loss gradient wrt embedding matrix E
        dL_dE = np.zeros_like(self.E)
        for i in range(t):
            dL_dE[chunked[i]] += dL_dX[i]

        return {"dE": dL_dE, "dWq": dL_dWq, "dWk": dL_dWk, "dWv": dL_dWv, "dW1": dL_dW1, "db1": dL_db1, "dW2": dL_dW2, "db2": dL_db2, "dWo": dL_dw0, "dbo": dL_db0}
