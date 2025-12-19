from train import train
from test import test

train()

while True:
    trainortest = input("Please enter '1' to train, and '2' to test.")

    if trainortest.index('1') != 0:
        examplenum = input("Please enter the number of examples you wish to train on.")
        train(int(examplenum))
    else:
                
    
    
    