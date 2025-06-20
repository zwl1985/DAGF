#!/bin/bash
# * laptop

# python ./train.py --model_name newbert --dataset laptop --lexicon laptop --seed 1000 --bert_lr 2e-5 --l2reg 1e-2 --num_epoch 50 --hidden_dim 768 --max_length 100 --cuda 0 --alpha 1.0 --beta 0.2 --gama 1.6

# * restaurant

# python ./train.py --model_name newbert --dataset restaurant --lexicon restaurant --seed 1000 --bert_lr 2e-5 --l2reg 1e-2 --num_epoch 30 --hidden_dim 768 --max_length 100 --cuda 0 --alpha 0.8 --beta 0.6 --gama 1.5

# * twitter

# python ./train.py --model_name newbert --dataset twitter --lexicon twitter --seed 1000 --bert_lr 1e-5 --l2reg 1e-4 --num_epoch 30 --hidden_dim 768 --max_length 100 --cuda 0 --alpha 0.4 --beta 0.3 --gama 1.3
