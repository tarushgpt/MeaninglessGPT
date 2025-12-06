import numpy as np

vocabulary = input("Please enter your vocabulary as a concatenated list (i.e.) \nabcdefghijklmnopqrstuvwxyz: \n\n")
example = input("Please enter your sample pattern for the transformer to learn on. Note: the characters must be contained within the vocabulary. Enter in the following format: \nabcd->e\n\n")

input_str = example.split("->")[0]
next_token = example.split("->")[1]

#vocabulary = list("abcdefghijklmnopqrstuvwxyz") #this is a built in test that you can run (the abcd->e example)

v_len = len(vocabulary) #helps for embedding matrix size
d_model = 8 #embedding length
d_k = 8 #key length
d_v = 8 #value length
d_ff = 16 #FFN length
E = np.random.normal(0, 0.02, (v_len, d_model)) #embedding matrix
Wq = np.random.normal(0, 0.02, (d_model, d_k)) #query weights
Wk = np.random.normal(0, 0.02, (d_model, d_k)) #key weights
Wv = np.random.normal(0, 0.02, (d_model, d_v)) #value weights
W1 = np.random.normal(0, 0.02, (d_model, d_ff)) #weight of first layer in FFN
b1 = np.random.normal(0, 0.02, (d_ff,)) #bias of first layer in FFN
W2 = np.random.normal(0, 0.02, (d_ff, d_model)) #weight of second layer in FFN
b2 = np.random.normal(0, 0.02, (d_model,)) #bias of second layer in FFN
Wo = np.random.normal(0, 0.02, (d_model, v_len)) #weight of output layer in FFN
bo = np.random.normal(0, 0.02, (v_len)) #weight of output bias in FFN
lr = 0.001 #learning rate for backward pass

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
    X = np.array(X) #dimensions of X are (t * d_model)

    #Attention Block

    Q = X @ Wq #dimensions of Wq are (d_model * d_k), so dimensions of Q are (t * d_k)
    K = X @ Wk #dimensions of Wk are (d_model * d_k), so dimensions of Q are (t * d_k)
    V = X @ Wv #dimensions of Wv are (d_model * d_v), so dimensions of Q are (t * d_v)

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

    scores_norm = scores / np.sqrt(d_k) + mask
    scores_soft = softmax(scores_norm) #still (t * t)

    output = scores_soft @ V #reminder that dimensions of V are (t * d_v), so output is (t * d_v)

    #FFN

    h1 = output @ W1 + b1 #dimensions of W1 is (d_v * dff) and dimensions of b1 is (d_ff,), so h1 is (t * d_ff)
    h2 = relu(h1) #dimensions do not change
    h3 = h2 @ W2 + b2 #dimensions of W2 is (d_ff * d_model) and dimensions of b2 is (d_model,), so h3 is (t, d_model)

    logits = h3[-1] @ Wo + bo #dimensions of W0 is (d_model, v_len) and b0 are (v_len,), so dimensions of logits are (v_len,)
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

    #
    dL_doutput = dL_dh1 @ W1.T

    #
    dL_dV = scores_soft.T @ dL_doutput
    dL_dscoressoft = dL_doutput @ V.T

    dL_dscores_norm = np.zeros_like(dL_dscoressoft)

    for i in range(t):
        p = scores_soft[i]         
        g = dL_dscoressoft[i]       
        dot = np.sum(g * p)
        dL_dscores_norm[i] = p * (g - dot)

    dL_dscores = 1/np.sqrt(d_k) * dL_dscores_norm
    dL_dQ = dL_dscores @ K
    dL_dKT = Q.T @ dL_dscores
    dL_dK = dL_dKT.T

    dL_dWv = X.T @ dL_dV
    dL_dWk = X.T @ dL_dK
    dL_dWq = X.T @ dL_dQ

    dL_dX = dL_dQ @ Wq.T + dL_dK @ Wk.T + dL_dV @ Wv.T
    dL_dE = np.zeros_like(E)
    for i in range(t):
        dL_dE[chunked[i]] += dL_dX[i]

    E -= lr * dL_dE
    Wq -= lr * dL_dWq
    Wk -= lr * dL_dWk
    Wv -= lr * dL_dWv
    W1 -= lr * dL_dW1
    b1 -= lr * dL_db1
    W2 -= lr * dL_dW2
    b2 -= lr * dL_db2
    Wo -= lr * dL_dw0
    bo -= lr * dL_db0

#printing weight matrices for clarity
print("E:\n", E, "\n")
print("Wq:\n", Wq, "\n")
print("Wk:\n", Wk, "\n")
print("Wv:\n", Wv, "\n")
print("W1:\n", W1, "\n")
print("b1:\n", b1, "\n")
print("W2:\n", W2, "\n")
print("b2:\n", b2, "\n")
print("Wo:\n", Wo, "\n")
print("bo:\n", bo, "\n")


chunked = []
for i in input_str: chunked.append(i)
for i in range(len(chunked)):
    chunked[i] = vocabulary.index(chunked[i])
t = len(chunked)

X = []
for i in range(t):
    index = chunked[i]
    X.append(E[index])
X = np.array(X) 

Q = X @ Wq 
K = X @ Wk 
V = X @ Wv

scores = Q @ K.T 
mask = []
for i in range(t):
    row = []
    for j in range(t):
        if j <= i: row.append(0.0)
        else: row.append(-np.inf)
    mask.append(row)
mask = np.array(mask) 

scores_norm = scores / np.sqrt(d_k) + mask
scores_soft = softmax(scores_norm) 

output = scores_soft @ V 

h1 = output @ W1 + b1 
h2 = relu(h1) 
h3 = h2 @ W2 + b2 

logits = h3[-1] @ Wo + bo 
probability = softmax_row(logits) 


print(vocabulary[np.argmax(probability)])

