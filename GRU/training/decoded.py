import sys
sys.path.append(".")
import torch
from src.data_until import config, data
from src.data_until.data import Vocab
from src.data_until.batcher import Example, Batch
from src.data_until.train_until import get_input_from_batch
from src.training.model import Model
from src.data_until.until import write_for_rouge

use_cuda = config.use_gpu and torch.cuda.is_available()

class Summary(object):
    def __init__(self, model_file_path ):
        self.vocab = Vocab(config.vocab_path, config.max_vocab_size)
        self.model = Model(self.vocab.size(), model_file_path, is_eval=True)

    def generate_one_summary(self, article_sent, abstract_sent):
        example = Example(article_sent, abstract_sent, self.vocab)
        batch = Batch([example], self.vocab, 1)

        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, context_t_1, coverage = \
            get_input_from_batch(batch,False)

        enc_outputs, enc_hidden_state = self.model.encoder(enc_batch, enc_lens)
        dec_input_t_1 = torch.LongTensor([self.vocab.get_index("<sos>")]).unsqueeze(1)
        dec_hidden_t_1 = enc_hidden_state

        output_ids = torch.LongTensor(config.max_dec_steps - 1, 1)
        for t in range(1, config.max_dec_steps):
            predictions, context_t, dec_hidden_t_1, attn_dist , next_coverage = self.model.decoder(
                                                                dec_input_t_1,
                                                                context_t_1,
                                                                dec_hidden_t_1,
                                                                enc_outputs,
                                                                enc_padding_mask,
                                                                extra_zeros,
                                                                enc_batch_extend_vocab,
                                                                coverage)
            topv, topi = predictions.topk(1)
            dec_input_t_1 = topi.detach()
            dec_input_t_1 = torch.LongTensor(
                [idx if idx < self.vocab.size() else self.vocab.get_index("<unk>") for idx in dec_input_t_1])\
                .view(dec_input_t_1.shape[0], 1)
            if topi.squeeze(1) == self.vocab.get_index("<eos>"):
                output_ids = output_ids[:t-1]
                break
            output_ids[t-1] = topi.squeeze(1)

        output_ids = output_ids.squeeze(1).cpu().numpy()
        output_ids = [id for id in output_ids if id != self.vocab.get_index("<pad>")]
        predict_words = data.outputidxs2words(output_ids, self.vocab, (batch.art_oovs[0] if config.pointer_gen else None))
        predict_words = ' '.join(predict_words)

        return predict_words

    def gen_summaries_of_val_test(self, val_article_sents, val_abstract_sents):
        print("-------- Start generate summary and write file to caculate rouge score --------------")
        decoded_list = []
        for i, (art_sent, abs_sent) in enumerate(zip(val_article_sents, val_abstract_sents)):
            decoded = self.generate_one_summary(art_sent, abs_sent)
            decoded_list.append(decoded)
        write_for_rouge(val_abstract_sents, decoded_list, config.model_dir, config.system_dir)
        print('------------------------------ finnish write file -----------------------------------')

article_sents, abstract_sents, pairs = data.readLanguage(config.data_train_example_path)
summary = Summary('model_1/model_50000_1589708798')
summary.gen_summaries_of_val_test(article_sents,abstract_sents)

