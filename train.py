import concurrent.futures
import numpy as np
import os
import config

class Train:

    def __init__(self):
        self.d_model = config.d_model
        self.d_k = config.d_k
        self.d_v = config.d_v
        self.d_ff = config.d_ff
        self.lr = config.lr

        if os.path.exists("weights/E.npy"):
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

        else:
            self.E = np.random.normal(0, 0.02, (self.v_len, self.d_model)) #embedding matrix
            self.Wq = np.random.normal(0, 0.02, (self.d_model, self.d_k)) #query weights
            self.Wk = np.random.normal(0, 0.02, (self.d_model, self.d_k)) #key weights
            self.Wv = np.random.normal(0, 0.02, (self.d_model, self.d_v)) #value weights
            self.W1 = np.random.normal(0, 0.02, (self.d_model, self.d_ff)) #weight of first layer in FFN
            self.b1 = np.random.normal(0, 0.02, (self.d_ff,)) #bias of first layer in FFN
            self.W2 = np.random.normal(0, 0.02, (self.d_ff, self.d_model)) #weight of second layer in FFN
            self.b2 = np.random.normal(0, 0.02, (self.d_model,)) #bias of second layer in FFN
            self.Wo = np.random.normal(0, 0.02, (self.d_model, self.v_len)) #weight of output layer in FFN
            self.bo = np.random.normal(0, 0.02, (self.v_len)) #weight of output bias in FFN


    def train(self, examplenum):
        
        vocabulary = input("Please enter your vocabulary as a concatenated list (i.e.) \nabcdefghijklmnopqrstuvwxyz: \n\n")

        self.v_len = len(vocabulary) 

        examples = []

        for i in range(examplenum):
            example = input("Please enter your sample pattern for the transformer to learn on. Note: the characters must be contained within the vocabulary. Enter in the following format: \nabcd->e\n\n")
            examples.append(example)

        with concurrent.futures.ThreadPoolExecutor(max_workers=examplenum) as executor:
            print("Training has begun on " + str(examplenum) + "patterns.")
            futures = []
            for example in examples:
                future = executor.submit(self.train_body(example, vocabulary))
                futures.append(future)

            futures = [future.result() for future in futures]

        E_list = []
        Wq_list = []
        Wk_list = []
        Wv_list = []
        W1_list = []
        b1_list = []
        W2_list = []
        b2_list = []
        Wo_list = []
        bo_list = []

        for trained in futures:
            E_list.append(trained["E"])
            Wq_list.append(trained["Wq"])
            Wk_list.append(trained["Wk"])
            Wv_list.append(trained["Wv"])
            W1_list.append(trained["W1"])
            b1_list.append(trained["b1"])
            W2_list.append(trained["W2"])
            b2_list.append(trained["b2"])
            Wo_list.append(trained["Wo"])
            bo_list.append(trained["bo"])

        np.save("weights/E.npy", E_list / len(E_list))
        np.save("weights/Wq.npy", Wq_list / len(Wq_list))
        np.save("weights/Wk.npy", Wk_list / len(Wk_list))
        np.save("weights/Wv.npy", Wv_list / len(Wv_list))
        np.save("weights/W1.npy", W1_list / len(W1_list))
        np.save("weights/b1.npy", b1_list / len(b1_list))



        print("Training completed. continuing on to testing.")



    def train_body(self, example, vocabulary):
        input_str = example.split("->")[0]
        next_token = example.split("->")[1]

        #softmax
        def softmax(n):
            soft_n = []
            for i  in range(len(n)):
                row = n[i]
                soft_row = softmax_row(row)
                soft_n.append(soft_row)
            return np.array(soft_n)

        #softmax of an individual row, subtracting max
        def softmax_row(n):
            row = n
            maxval = np.max(row)
            exp = np.exp(row - maxval)
            soft_n = exp / np.sum(exp)
            return soft_n

        #relu
        def relu(n):
            relu_n = []
            for i in range(len(n)):
                n_row = []
                for j in range(len(n[i])):
                    n_row.append(max(n[i][j], 0))
                relu_n.append(n_row)
            return np.array(relu_n)

        #derivative of relu for backpropogation
        def reluprime(n):
            relu_prime = np.zeros_like(n)
            for i in range(len(relu_prime)):
                for j in range(len(relu_prime[i])):
                    if n[i][j] > 0: relu_prime[i][j] = 1
                    else: relu_prime[i][j] = 0
            return relu_prime

        #running 1000 steps for training

        for step in range(1000):
            
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
            scores_soft = softmax(scores_norm) #still (t * t)

            output = scores_soft @ V #reminder that dimensions of V are (t * self.d_v), so output is (t * self.d_v)

            #FFN

            h1 = output @ W1 + b1 #dimensions of W1 is (self.d_v * dff) and dimensions of b1 is (self.d_ff,), so h1 is (t * self.d_ff)
            h2 = relu(h1) #dimensions do not change
            h3 = h2 @ W2 + b2 #dimensions of W2 is (self.d_ff * self.d_model) and dimensions of b2 is (self.d_model,), so h3 is (t, self.d_model)

            logits = h3[-1] @ Wo + bo #dimensions of W0 is (self.d_model, self.v_len) and b0 are (self.v_len,), so dimensions of logits are (self.v_len,)
            probability = softmax_row(logits) #dimensions do not change

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
            dL_dh1 = dL_dh2 * reluprime(h1)
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
            dL_dE = np.zeros_like(E)
            for i in range(t):
                dL_dE[chunked[i]] += dL_dX[i]

            #weight updates
            E -= self.lr * dL_dE
            Wq -= self.lr * dL_dWq
            Wk -= self.lr * dL_dWk
            Wv -= self.lr * dL_dWv
            W1 -= self.lr * dL_dW1
            b1 -= self.lr * dL_db1
            W2 -= self.lr * dL_dW2
            b2 -= self.lr * dL_db2
            Wo -= self.lr * dL_dw0
            bo -= self.lr * dL_db0

            return {"E": E, "Wq": Wq, "Wk": Wk, "Wv": Wv, "W1": W1, "b1": b1, "W2": W2, "b2": b2, "Wo": Wo, "bo": bo}
