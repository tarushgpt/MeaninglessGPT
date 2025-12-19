from train import Train
from test import Test
import config, os
import numpy as np


def weight_initialize(vocabulary):
    v_len = len(vocabulary)

    E = np.random.normal(0, 0.02, (v_len, config.d_model)) #embedding matrix
    Wq = np.random.normal(0, 0.02, (config.d_model, config.d_k)) #query weights
    Wk = np.random.normal(0, 0.02, (config.d_model, config.d_k)) #key weights
    Wv = np.random.normal(0, 0.02, (config.d_model, config.d_v)) #value weights
    W1 = np.random.normal(0, 0.02, (config.d_model, config.d_ff)) #weight of first layer in FFN
    b1 = np.random.normal(0, 0.02, (config.d_ff,)) #bias of first layer in FFN
    W2 = np.random.normal(0, 0.02, (config.d_ff, config.d_model)) #weight of second layer in FFN
    b2 = np.random.normal(0, 0.02, (config.d_model,)) #bias of second layer in FFN
    Wo = np.random.normal(0, 0.02, (config.d_model, v_len)) #weight of output layer in FFN
    bo = np.random.normal(0, 0.02, (v_len)) #weight of output bias in FFN

    np.save("weights/E.npy", E)
    np.save("weights/Wq.npy", Wq)
    np.save("weights/Wk.npy", Wk)
    np.save("weights/Wv.npy", Wv)
    np.save("weights/W1.npy", W1)
    np.save("weights/b1.npy", b1)
    np.save("weights/W2.npy", W2)
    np.save("weights/b2.npy", b2)
    np.save("weights/Wo.npy", Wo)
    np.save("weights/bo.npy", bo)

def main():

    newornot = input("Welcome to MeaninglessGPT! If you are new, press 'Enter.' If not, press any other key. ")
    print("\n")

    
    if newornot == "":
        print("This model uses a manual backpropogation system (no PyTorch, just NumPy) as a practice implementation of a transformer.\n")
        print("It is entirely designed to overfit on specific examples, hence the name 'MeaninglessGPT.'\n")
        print("All weights will be saved in ~/weights/. You can adapt the config file to change model characteristics.\n")
        print("Please note that since MeaninglessGPT overfits on patterns, it has virtually no memory from past examples. In other words, your old examples will be effectively overwritten, although some semblance of memory might remain.\n")
        print("Happy experimenting! -Tarush\n\n")

    

    while True:
        trainortest = input("Please enter '1' to train, '2' to test, or '3' to exit. ")
        if trainortest == "1":
            vocabulary = input("Please enter your vocabulary as a concatenated list. ")

            if not os.path.exists("weights/E.npy"):
                weight_initialize(vocabulary)

            train = Train()

            examplenum = input("Please enter the number of examples you wish to train on. ")
            train.train(int(examplenum), vocabulary)
        elif trainortest == "2":
            if vocabulary:
                test = Test()
                test.test(vocabulary)
            else:
                print("You have not initialized a vocabulary. Please train on a pattern first and try again!\n")
        else:
            break

main()  
        