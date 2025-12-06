import numpy as np

#printing final weight matrices for clarity

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

