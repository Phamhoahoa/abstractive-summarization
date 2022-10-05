import sys
sys.path.append(".")
import torch
from src.data_until import config, data
from src.data_until.data import Vocab
from src.data_until.batcher import Batcher
from src.training.model import Model
import time
import os
from src.data_until.train_until import get_input_from_batch, get_output_from_batch

use_cuda = config.use_gpu and torch.cuda.is_available()

class Evaluate(object):
    def __init__(self, model_file_path):
        self.article_sents, self.abstract_sents, self.pairs = data.readLanguage(config.data_test_path)
        self.vocab = Vocab(config.vocab_path, config.max_vocab_size)

        self.batcher = Batcher(self.article_sents, self.abstract_sents, self.vocab)
        print('Số batch là : ', len(self.batcher.n_batchs))
        self.model = Model(self.vocab.size(), model_file_path, is_eval=True)

    def eval_one_batch(self, batch):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros,context_t_1, coverage= \
            get_input_from_batch(batch,use_cuda)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, use_cuda)

        enc_outputs, enc_hidden_state = self.model.encoder(enc_batch, enc_lens)
        dec_hidden_t_1 = enc_hidden_state

        step_losses = []
        for t in range(min(max_dec_len, config.max_dec_steps)):
            dec_input_t_1 = dec_batch[:, t].unsqueeze(1)  # Teacher forcing
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

        return loss.item()

    def run_eval(self):
        batchs = self.batcher.n_batchs
        iter = 0
        total_loss = 0
        start = time.time()
        while iter < len(self.batcher.n_batchs):
            batch = batchs[iter]
            batch_loss = self.eval_one_batch(batch)
            total_loss += batch_loss
            iter += 1
            print_interval = 100
            if iter % print_interval == 0:
                  print('Batch {} Loss {:.4f} , time {:.4f}'.format(iter,  batch_loss, time.time() - start))
                  start = time.time()

        avg_loss = total_loss / len(self.batcher.n_batchs)
        print('Avg Loss {:.4f}'.format( avg_loss))
        return avg_loss

if __name__ == '__main__':
    model_dir = os.path.join(config.root_dir, 'model_1/')
    model_name = 'model_30000_1589702763'

    model_file_path = model_dir + model_name
    eval_processor = Evaluate(model_file_path)
    eval_processor.run_eval()


