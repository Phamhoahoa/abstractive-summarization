import sys
sys.path.append(".")
import torch
from src.data_until  import config, data
from src.data_until.data import Vocab
from src.data_until.batcher import Batcher
from src.training.model import Model
from torch import optim
import time
import os
from src.data_until.train_until import get_input_from_batch, get_output_from_batch

use_cuda = config.use_gpu and torch.cuda.is_available()

class Train(object):
    def __init__(self):
        self.article_sents, self.abstract_sents, self.pairs = data.readLanguage(config.data_train_path)
        self.vocab = Vocab(config.vocab_path, config.max_vocab_size)

        self.batcher = Batcher(self.article_sents, self.abstract_sents, self.vocab)
        print('Số batch là : ',len(self.batcher.n_batchs))

        self.model_dir = os.path.join(config.root_dir, 'model_1')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

    def save_model(self, loss_iters, iter):
        state = {
            'iter': iter,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss_iters': loss_iters
        }
        model_save_path = os.path.join(self.model_dir, 'model_%d_%d' % (iter, int(time.time())))
        torch.save(state, model_save_path)

    def setup_train(self, model_file_path=None):
        self.model = Model(self.vocab.size(), model_file_path)
        print(self.model.encoder)
        print(self.model.decoder)

        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters())
        # self.optimizer = Adagrad(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)
        self.optimizer = optim.Adam(params, config.lr)

        start_iter = 0
        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            self.optimizer.load_state_dict(state['optimizer'])
            start_iter = state['iter']
            current_loss = state['loss_iters']
            print('Tiếp tục iter {}'.format(start_iter))
            print('Loss {}\n'.format(current_loss[-1]))
        return start_iter

    def train_one_batch(self, batch, clip=1):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, context_t_1, coverage = \
            get_input_from_batch(batch,use_cuda)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, use_cuda)

        self.optimizer.zero_grad()

        enc_outputs, enc_hidden_state = self.model.encoder(enc_batch, enc_lens)
        dec_hidden_t_1 = enc_hidden_state
        step_losses = []
        for t in range(min(max_dec_len, config.max_dec_steps)):
            dec_input_t_1 = dec_batch[:, t].unsqueeze(1) # Teacher forcing
            predictions, context_t, dec_hidden_t_1, attn_dist , next_coverage = self.model.decoder(
                                                                dec_input_t_1,
                                                                context_t_1,
                                                                dec_hidden_t_1,
                                                                enc_outputs,
                                                                enc_padding_mask,
                                                                extra_zeros,
                                                                enc_batch_extend_vocab,
                                                                coverage)
            target = target_batch[:, t]
            gold_probs = torch.gather(predictions, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps)
            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage
            step_mask = dec_padding_mask[:, t]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)
        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses / dec_lens_var
        loss = torch.mean(batch_avg_loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(self.model.decoder.parameters(), clip)
        self.optimizer.step()

        return loss.item()

    def trainIters(self, n_iters, model_file_path=None):
        iter  = self.setup_train(model_file_path)
        loss_iters = []
        start = time.time()
        batchs = self.batcher.n_batchs
        i = 0
        while iter < n_iters: 
            batch = batchs[i]
            loss = self.train_one_batch(batch)
            # loss_iters.append(loss)
            loss_iters.append(loss)
            iter += 1
            print_interval = 50
            if iter % print_interval == 0:
                  print('Batch {} :  Loss {:.4f} - time {:.4f}'.format(iter,  loss, time.time() - start))
                  start = time.time()
            if iter % 500 == 0:
                self.save_model(loss_iters, iter)
            i += 1
            if i == len(batchs):
                i = 0


if __name__ == '__main__':
    train_processor = Train()

    # model_file_name = 'model_8000_1588606000'
    # model_file_path = model_dir + model_file_name
    # old best loss without teacher forcing là 0.2897
    train_processor.trainIters(config.max_iterations)
