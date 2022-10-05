import sys
sys.path.append(".")
from torch.utils.data import Dataset, DataLoader
from src.data_until import config, data
import numpy as np

class Example(object):

    def __init__(self, article, abstract, vocab):
        # Get ids of special tokens
        start_decoding = vocab.get_index("<sos>")
        stop_decoding = vocab.get_index("<eos>")
        # Process the article
        article_words = article.split()
        self.enc_len = len(article_words)  # store the length after truncation but before padding
        self.enc_input = [vocab.get_index(w) for w in article_words]  # list of word ids; OOVs are represented by the id for UNK token

        # Process the abstract
        abstract_words = abstract.split()  # list of strings
        abs_ids = [vocab.get_index(w) for w in abstract_words]
        self.dec_input, self.target = self.get_dec_inp_targ_seqs(abs_ids, config.max_dec_steps, start_decoding, stop_decoding)
        self.dec_len = len(self.dec_input)

        # If using pointer-generator mode, we need to store some extra info
        if config.pointer_gen:
            # Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id; also store the in-article OOVs words themselves
            self.enc_input_extend_vocab, self.article_oovs = data.article2idxs(article_words, vocab)
            # Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
            abs_ids_extend_vocab = data.abstract2idxs(abstract_words, vocab, self.article_oovs)
            # Overwrite decoded target sequence so it uses the temp article OOV ids
            _, self.target = self.get_dec_inp_targ_seqs(abs_ids_extend_vocab, config.max_dec_steps, start_decoding, stop_decoding) # end token

    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len:  # truncate
            inp = inp[:max_len]
            target = target[:max_len]  # no end_token
        else:  # no truncation
            target.append(stop_id)  # end token
        assert len(inp) == len(target)
        return inp, target

    def pad_decoder_inp_targ(self, max_len, pad_id):
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)

    def pad_encoder_input(self, max_len, pad_id):
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)
        if config.pointer_gen:
            while len(self.enc_input_extend_vocab) < max_len:
                self.enc_input_extend_vocab.append(pad_id)


class Batch(object):
    def __init__(self, example_list, vocab, batch_size):
        self.batch_size = batch_size
        self.pad_id = vocab.get_index("<pad>")  # id of the PAD token used to pad sequences
        self.init_encoder_seq(example_list)  # initialize the input to the encoder
        self.init_decoder_seq(example_list)  # initialize the input and targets for the decoded

    def init_encoder_seq(self, example_list):
        # Determine the maximum length of the encoder input sequence in this batch
        max_enc_seq_len = max([ex.enc_len for ex in example_list])

        # Pad the encoder input sequences up to the length of the longest sequence
        for ex in example_list:
            ex.pad_encoder_input(max_enc_seq_len, self.pad_id)

        # Initialize the numpy arrays
        # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
        self.enc_batch = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
        self.enc_lens = np.zeros((self.batch_size), dtype=np.int32)
        self.enc_padding_mask = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.float32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.enc_batch[i, :] = ex.enc_input[:]
            self.enc_lens[i] = ex.enc_len
            for j in range(ex.enc_len):
                self.enc_padding_mask[i][j] = 1

        # For pointer-generator mode, need to store some extra info
        if config.pointer_gen:
            # Determine the max number of in-article OOVs in this batch
            self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
            # Store the in-article OOVs themselves
            self.art_oovs = [ex.article_oovs for ex in example_list]
            # Store the version of the enc_batch that uses the article OOV ids
            self.enc_batch_extend_vocab = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
            for i, ex in enumerate(example_list):
                self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]

    def init_decoder_seq(self, example_list):
        # Pad the inputs and targets
        for ex in example_list:
            ex.pad_decoder_inp_targ(config.max_dec_steps, self.pad_id)

        # Initialize the numpy arrays.
        self.dec_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
        self.dec_lens = np.zeros((self.batch_size), dtype=np.int32)
        self.target_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
        self.dec_padding_mask = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.float32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.dec_batch[i, :] = ex.dec_input[:]
            self.target_batch[i, :] = ex.target[:]
            self.dec_lens[i] = ex.dec_len
            for j in range(ex.dec_len):
                self.dec_padding_mask[i][j] = 1


# tạo batch cho bước train
class MyData(Dataset):
    def __init__(self, article_sents, abstract_sents):
        self.article_sents = article_sents
        self.abstract_sents = abstract_sents

    def __getitem__(self, index):
        article_sents = self.article_sents[index]
        abstract_sents = self.abstract_sents[index]
        return article_sents, abstract_sents

    def __len__(self):
        return len(self.article_sents)


class Batcher(object):
    def __init__(self, article_sents, abstract_sents, vocab):
        self.article_sents = article_sents
        self.abstract_sents = abstract_sents
        self.vocab = vocab
        self.n_batchs = list()
        self.batchs()

    def batchs(self):
        dataset = MyData(self.article_sents, self.abstract_sents)
        train_set = DataLoader(dataset, batch_size=config.batch_size,
                               drop_last=True,
                               shuffle=True)

        for (batch, (article_sents, abstract_sents)) in enumerate(train_set):
            batch_example = []
            for (i, (article, abstract)) in enumerate(zip(article_sents, abstract_sents)):
                example = Example(article, abstract, self.vocab)
                batch_example.append(example)
            batch = Batch(batch_example, self.vocab, config.batch_size)
            self.n_batchs.append(batch)

    def next_batch(self):
        count = 0
        while count < len(self.n_batchs):
            batch = self.n_batchs[count]
            count  +=  1
            if count == len(self.n_batchs):
                count = 0
            return batch
            
            
