import os

root_dir = os.getcwd()
print(root_dir)

vocab_path = 'data/new_vietnamese_vocab.txt'
pre_embedding_path = 'data/news_word_normalize_model.bin'
data_train_example_path = './data/test.csv'
data_train_path = './data/train_tokenized.csv'
data_val_path ='./data/val_tokenized.csv'
data_test_path = './data/test_tokenized.csv'

system_dir = 'decoded/'
model_dir = 'reference/'

hidden_dim= 512
# hidden_dim= 256
emb_dim= 256
batch_size= 16 

max_enc_steps=80
max_dec_steps=64
max_vocab_size=300000
min_count = 1
max_iterations = 50000
pointer_gen = False
is_coverage = False

eps = 1e-12
lr=0.001
enc_dropout = 0.1
# enc_dropout = 0.2
dec_dropout = 0.1
# dec_dropout = 0.2
cov_loss_wt = 1.0
use_gpu=False

