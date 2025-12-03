# Meaningless GPT

## Overview

Implements a single head attention, 914 parameter micro GPT using a zero-PyTorch, NumPy first approach to learn a singular pattern "abcd" -> "e". 

Uses a character based tokenizer and Gaussian initialized embeddings and weight matrices. Does not use positionals or LayerNorm- please see AureliusGPT for a full, multi head attention, ~700k parameter model with sinusoidal initialized positional matrices and LayerNorm, with 3 transformer blocks.

Important note: this GPT is completely "meaningless," hence the name: it is entirely designed to overfit on a specific example and output the correct token given the exact same input string and fails in any other test. 

It was intended for my personal learning and practice through a simple version of the forward and backward pass of a smallscale single head transformer, and is a partial and miniscule reimplementation of "Attention is All You Need (Vaswani et al.)." 


To run the project, use:
```
uv pip install -r requirements.txt
python transformer.py
```