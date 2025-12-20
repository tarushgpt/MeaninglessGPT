# Meaningless GPT

## Overview

MeaninglessGPT mplements a single head attention micro GPT using a zero-PyTorch, NumPy first approach to learn a singular pattern. 

It uses a character based tokenizer and Gaussian initialized embeddings and weight matrices. On the other hand, it does not use positionals or LayerNorm- please see AureliusGPT for a full, multi head attention, ~700k parameter model with sinusoidal initialized positional matrices and LayerNorm, with 3 transformer blocks.

Important note: this GPT is completely "meaningless," hence the name: it is entirely designed to overfit on a specific example and output the correct token given the exact same input string and fails in any other test. 

It was intended for my personal learning and practice through a simple version of the forward and backward pass of a smallscale single head transformer, and is a partial and miniscule reimplementation of "Attention is All You Need (Vaswani et al.)." 

I attempted Federated Averaging, Synchronous Parallel SGD, and Gradient Accumulation / Batch Descent in MeaninglessGPT, but these methods were far too violent for a model this small, designed for overfitting. In previous commits, my implementations for some of these batching strategies are visible. 


To run the project, use:
```
uv pip install -r requirements.txt
python main.py
```