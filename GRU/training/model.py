from __future__ import unicode_literals, print_function, division
from src.data_until import config
import fasttext as ft
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import sys
sys.path.append(".")

use_cuda = config.use_gpu and torch.cuda.is_available()

class Encoder(nn.Module):
    def __init__(self, embedding, hidden_dim, dropout=config.enc_dropout):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = torch.nn.Embedding.from_pretrained(embedding)
        self.embedding_dim = self.embedding.embedding_dim
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(self.embedding_dim,
                          hidden_dim,
                          batch_first=True,
                          bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x, x_lens, hidden=None):
        embedded = self.dropout(self.embedding(x))

        packed = pack_padded_sequence(embedded, x_lens, batch_first=True, enforce_sorted= False)

        outputs, hidden = self.gru(packed, hidden)

        outputs, _ = pad_packed_sequence(outputs)

        enc_outputs = outputs[:, :, :self.hidden_dim] + outputs[:, :, self.hidden_dim:]
        last_hidden_state = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))).unsqueeze(0)

        return enc_outputs, last_hidden_state

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        if config.is_coverage:
            self.W_c = nn.Linear(1, hidden_dim, bias=False)
        self.Wa = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Ua = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.va = nn.Linear(hidden_dim, 1, bias = False)

    def forward(self, hidden, encoder_outputs, enc_padding_mask, coverage):
        attn_weights = self.score(hidden, encoder_outputs, coverage)  # [B, T]
        attn_dists_ = torch.softmax(attn_weights, dim=1) * enc_padding_mask
        normalization_factor = attn_dists_.sum(1, keepdim=True)
        attn_dists = attn_dists_ / normalization_factor
        attn_dists = attn_dists.unsqueeze(1)  # [B, 1, T]

        if config.is_coverage:
            coverage = coverage.unsqueeze(1) + attn_dists
            coverage = coverage.squeeze(1)

        return attn_dists , coverage # attn_dists : [B, 1, T], coverage : [B,T]

    def score(self, last_hidden, encoder_outputs, coverage):
        x = last_hidden.unsqueeze(1) # => x : (B, 1, H)
        att_features = self.Wa(x) + self.Ua(encoder_outputs)
        if config.is_coverage:
            coverage_input = coverage.unsqueeze(2)  # [B, T, 1]
            coverage_feature = self.W_c(coverage_input)  # [B, T , H]
            att_features = att_features + coverage_feature # [B, T, H]
        energy = torch.tanh(att_features) # => energy : [B, T, H]
        return self.va(energy).squeeze(2) # [B, T, H] x [H, 1] = [B, T, 1] => [B, T]

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding, hidden_dim, dropout=config.dec_dropout):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_dim
        self.embedding = torch.nn.Embedding.from_pretrained(embedding)
        self.embedding_dim = self.embedding.embedding_dim
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.gru = nn.GRU(self.hidden_size + self.embedding_dim, self.hidden_size, batch_first=True)
        self.attention = Attention(self.hidden_size)

        self.out = nn.Linear(2*self.hidden_size , self.hidden_size)
        self.fc_out = nn.Linear(self.hidden_size, self.vocab_size)

        if config.pointer_gen:
            # khởi tạo cho point-gen
            self.Wc = nn.Linear(hidden_dim, 1)
            self.Wh = nn.Linear(hidden_dim, 1)
            self.Wx = nn.Linear(self.embedding_dim + hidden_dim, 1)

    def forward(self, y_t_1, context_t_1, hidden_t_1, encoder_outputs, enc_padding_mask, extra_zeros, enc_batch_extend_vocab, coverage_t):

        # x shape sau khi qua embedding == (B, 1, E) ( đầu vào chỉ là 1 token)
        embedded_t_1 = self.dropout(self.embedding(y_t_1))

        # attention shape sau khi concatenation(x, context_vector)== (B, 1, E + H)
        rnn_input_t = torch.cat((context_t_1.unsqueeze(1), embedded_t_1), 2)

        # pass qua GRU
        output_t, hidden_t = self.gru(rnn_input_t, hidden_t_1)  # hidden == (1, B, H), output == (B,1, H)

        # enc_outputs == (T, B, H) => permute(1,0,2)  (B, T, H)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        attn_dists_t , coverage_next = self.attention(hidden_t.squeeze(0), encoder_outputs, enc_padding_mask, coverage_t)  # => (B, 1, T)

        context_t = attn_dists_t.bmm(encoder_outputs)  # => ( B, 1, H) = (B,1,T)X(B, T, H)

        output_t = self.out(torch.cat((output_t, context_t), dim = 2))
        output_t = self.fc_out(output_t.squeeze(1))
        vocab_dist_t = torch.softmax(output_t, dim=1)  # => [B, O]

        context_t = context_t.squeeze(1)
        if config.pointer_gen:
            input_gen = self.Wc(context_t) + self.Wh(hidden_t.squeeze(0)) + self.Wx(rnn_input_t.squeeze(1))
            p_gen = torch.sigmoid(input_gen)  # [B, 1]

            attn_dists_t = attn_dists_t.squeeze(1)
            vocab_dist_t_ = p_gen * vocab_dist_t  # => vocab_dist_ : [B, O]
            attn_dists_t_ = (1 - p_gen) * attn_dists_t  # attn_weights : [B, T]

            if extra_zeros is not None:
                vocab_dist_t_ = torch.cat([vocab_dist_t_, extra_zeros], 1)  # vocab_dist_ => [B, O_size + OOV_size]
            final_dist_t = vocab_dist_t_.scatter_add(1, enc_batch_extend_vocab, attn_dists_t_)
        else:
            final_dist_t = vocab_dist_t

        return final_dist_t, context_t, hidden_t, attn_dists_t, coverage_next


class Model(object):
    def __init__(self, vocab_size, model_file_path=None, is_eval=False):
        pre_embedding = ft.load_model(config.pre_embedding_path)
        weight_embedding_matrix = torch.FloatTensor(pre_embedding.get_input_matrix())
        encoder = Encoder(weight_embedding_matrix, config.hidden_dim)
        decoder = Decoder(vocab_size,weight_embedding_matrix , config.hidden_dim)
        # shared the embedding between encoder and decoded
        decoder.embedding.weight = encoder.embedding.weight

        if is_eval:
            encoder = encoder.eval()
            decoder = decoder.eval()

        if use_cuda:
            encoder = encoder.cuda()
            decoder = decoder.cuda()

        self.encoder = encoder
        self.decoder = decoder

        if model_file_path is not None:
            if torch.cuda.is_available():
                # map_location = lambda storage, loc: storage.cuda()
                state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            else:
                # map_location = 'cpu'
                state = torch.load(model_file_path, map_location='cpu')
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)
