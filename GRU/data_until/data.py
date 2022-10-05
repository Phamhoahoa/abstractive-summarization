# Default word tokens
PAD_TOKEN = '<pad>'
UNKNOWN_TOKEN = '<unk>'
START_DECODING = '<sos>'
STOP_DECODING = '<eos>'
import pandas as pd

def readLanguage(filename):
    df = pd.read_csv(filename)
    input_lang = []
    target_lang = []
    pairs = []
    for ind, val in df.iterrows():
        input_lang.append(val['original'])
        target_lang.append(val['summary'])
        pairs.append([val['original'], val['summary']])
    print("Dữ liệu có %s cặp câu" % len(pairs))
 
    return input_lang, target_lang, pairs

class Vocab:
    def __init__(self, vocab_file, max_size):
        self.word2index = {}
        self.index2word = {}
        self._count = 0  # keeps track of total number of words in the Vocab

        # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
        for w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
            self.word2index[w] = self._count
            self.index2word[self._count] = w
            self._count += 1

        # Read the vocab file and add words up to max_size
        with open(vocab_file, 'r') as vocab_f:
            lines = vocab_f.readlines()
            for line in lines:
                w = line.strip()
                if w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                    raise Exception('[UNK], [PAD], [START] and [STOP] không nên có trong vocab, nhưng xuất hiện %s trong vocab_file' % w)
                if w in self.word2index:
                     print('Từ lặp lại trong vocabulary file: %s' % w)
                     continue
                self.word2index[w] = self._count
                self.index2word[self._count] = w
                self._count += 1
                if max_size != 0 and self._count >= max_size:
                    print("max_size của vocab l à%i; đã có %i từ. dừng reading." % (
                    max_size, self._count))
                    break

        print("Kết thúc xây dựng vocabulary tổng số %i từ. Từ cuối cùng đươc thêm vào : %s" % (
        self._count, self.index2word[self._count - 1]))

    def get_index(self, word):
        if word not in self.word2index:
            return self.word2index["<unk>"]
        return self.word2index[word]

    def get_word(self, word_id):
        if word_id not in self.index2word:
            raise ValueError('Id không tìm thấy trong vocab: %d' % word_id)
        return self.index2word[word_id]

    def size(self):
        return self._count



def article2idxs(article_words, vocab):
    ids = []
    oovs = []
    unk_id = vocab.word2index["<unk>"]
    for w in article_words:
        i = vocab.get_index(w)
        if i == unk_id:  # If w is OOV
            if w not in oovs:  # Add to list of OOVs
                oovs.append(w)
            oov_num = oovs.index(w)  # This is 0 for the first article OOV, 1 for the second article OOV...
            ids.append(vocab.size() + oov_num)  # This is e.g. 50000 for the first article OOV, 50001 for the second...
        else:
            ids.append(i)
    return ids, oovs

def abstract2idxs(abstract_words, vocab, article_oovs):
    ids = []
    unk_id = vocab.word2index["<unk>"]
    for w in abstract_words:
        i = vocab.get_index(w)
        if i == unk_id:  # If w is an OOV word
            if w in article_oovs:  # If w is an in-article OOV
                vocab_idx = vocab.size() + article_oovs.index(w)  # Map to its temporary article OOV number
                ids.append(vocab_idx)
            else:  # If w is an out-of-article OOV
                ids.append(unk_id)  # Map to the UNK token id
        else:
            ids.append(i)
    return ids

def outputidxs2words(id_list, vocab, article_oovs):
    words = []
    for i in id_list:
        try:
            w = vocab.get_word(i) # might be [UNK]
        except ValueError as e: # w is OOV
            assert article_oovs is not None, "Error: id của từ không thuộc vocab, có thể trong OOV"
            article_oov_idx = i - vocab.size()
            try:
                w = article_oovs[article_oov_idx]
            except ValueError as e: # i doesn't correspond to an article oov
                raise ValueError('Error: từ không có trong OOV')
        words.append(w)
    return words

def show_art_oovs(article, vocab):
    unk_token = vocab.get_index("<unk>")
    words = article.split(' ')
    words = [("__%s__" % w) if vocab.get_index(w)==unk_token else w for w in words]
    out_str = ' '.join(words)
    return out_str

def show_abs_oovs(abstract, vocab, article_oovs):
    print(article_oovs)
    unk_token = vocab.word2index["<unk>"]
    words = abstract.split(' ')
    new_words = []
    for w in words:
        if vocab.get_index(w) == unk_token: # w is oov
            if article_oovs is None: # baseline mode
                new_words.append("__%s__" % w)
            else: # pointer-generator mode
                if w in article_oovs:
                    new_words.append("__%s__" % w)
                else:
                    new_words.append("!!__%s__!!" % w)
        else: # w is in-vocab word
            new_words.append(w)
        out_str = ' '.join(new_words)
    return out_str
